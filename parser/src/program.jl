#= General conic linear (convex) optimization problem.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington),
                   and Autonomous Systems Laboratory (Stanford University)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

if isdefined(@__MODULE__, :LanguageServer)
    include("../../utils/src/Utils.jl")
    include("function.jl")
    include("cone.jl")
    using .Utils
end

using JuMP
using ECOS

import JuMP: value

export ConicProgram, blocks, variable!, value
export @new_variable, @new_parameter

const AtomicVariable = VariableRef
const AtomicConstant = Float64
const AtomicArgument = Union{AtomicVariable, AtomicConstant}
const BlockValue{T,N} = AbstractArray{T,N}
const LocationIndices = Array{Int}

abstract type AbstractArgument{T<:AtomicArgument} end
abstract type AbstractConicProgram end

""" A block in the overall argument. """
mutable struct ArgumentBlock{T<:AtomicArgument, N} <: AbstractArray{T, N}
    value::BlockValue{T, N}       # The value of the block
    name::String                  # Argument name
    blid::Int                     # Block number in the argument
    elid::LocationIndices         # Element indices in the argument
    arg::Ref{AbstractArgument{T}} # Reference to owning argument

    """
        ArgumentBlock(arg, shape, blid, elid1, name)

    Basic constructor for a contiguous block in the stacked argument.

    # Arguments
    - `arg`: reference to the parent argument.
    - `shape`: the block shape.
    - `blid`: block location in the parent argument.
    - `elid1`: element index for the first element of block.
    - `name`: block name.

    # Returns
    - `blk`: a new argument block object.
    """
    function ArgumentBlock(
        arg::AbstractArgument{T},
        shape::NTuple{N, Int},
        blid::Int,
        elid1::Int,
        name::String)::ArgumentBlock{T, N} where {T<:AtomicArgument, N}

        # Initialize the value
        if T<:AtomicVariable
            value = Array{AtomicVariable, N}(undef, shape...)
            populate!(value, arg.prog[].mdl; name=name)
        else
            value = fill(NaN, shape...)
        end

        # Building indices
        elid = (1:length(value)).+(elid1-1)

        arg_ptr = Ref{AbstractArgument{T}}(arg)
        blk = new{T, N}(value, name, blid, elid, arg_ptr)

        return blk
    end # function

    """
        ArgumentBlock(block, Id...[; link])

    A kind of "move constructor" for slicing a block.

    # Arguments
    - `block`: the original block.
    - `Id...`: the slicing indices/colons.

    # Keywords
    - `link`: (optional) whether to return a view instead of a regular
      slice. If not, an array is returned (which may produce a copy).

    # Returns
    - `sliced_block`: the sliced block.
    """
    function ArgumentBlock(
        block::ArgumentBlock{T, N},
        Id...; link::Bool=false)::ArgumentBlock{T} where {T<:AtomicArgument, N}

        # Slice the block value
        sliced_value = (link) ? view(block.value, Id...) : block.value[Id...]
        if !(typeof(sliced_value)<:AbstractArray)
            sliced_value = fill(sliced_value, ())
        end

        # Get the element indices for the slice (which are a subset of the
        # original)
        sliced_elid = block.elid[LinearIndices(block.value)[Id...]]
        if !(typeof(sliced_elid)<:AbstractArray)
            sliced_elid = fill(sliced_elid, ())
        end

        K = ndims(sliced_elid)
        sliced_block = new{T, K}(sliced_value, block.name, block.blid,
                                 sliced_elid, block.arg)

        return sliced_block
    end # function
end # struct

"""
Provide an interface to make `ArgumentBlock` behave like an array. See the
[documentation](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
"""
Base.size(blk::ArgumentBlock) = size(blk.value)
Base.getindex(blk::ArgumentBlock, I...) = ArgumentBlock(blk, I...)
Base.view(blk::ArgumentBlock, I...) = ArgumentBlock(blk, I...; link=true)
Base.setindex!(blk::ArgumentBlock{T}, v::T, i::Int) where T = blk.value[i] = v
Base.setindex!(blk::ArgumentBlock{T}, v::T,
               I::Vararg{Int, N}) where {T, N} = blk.value[I...] = v

const BlockBroadcastStyle = Broadcast.ArrayStyle{ArgumentBlock}
Broadcast.BroadcastStyle(::Type{<:ArgumentBlock}) = BlockBroadcastStyle()
Broadcast.broadcastable(blk::ArgumentBlock) = blk

const ArgumentBlocks{T} = Vector{ArgumentBlock{T}}

""" Stacked function argument. """
mutable struct Argument{T<:AtomicArgument} <: AbstractArgument{T}
    blocks::ArgumentBlocks{T}       # The individual argument blocks
    numel::Int                      # Number of atomic elements
    prog::Ref{AbstractConicProgram} # The parent conic program

    """
        Argument{T}()

    Empty constructor.

    # Returns
    - `arg`: the argument object.
    """
    function Argument{T}()::Argument{T} where {T<:AtomicArgument}

        # Empty initial values
        numel = 0
        blocks = ArgumentBlocks{T}(undef, 0)
        prog = Ref{AbstractConicProgram}()

        arg = new{T}(blocks, numel, prog)

        return arg
    end # function
end # struct

# Some helper functions for Argument
blocks(arg::Argument) = length(arg.blocks)

"""
Support some array operations for Argument.
"""
Base.length(arg::Argument) = arg.numel
Base.getindex(arg::Argument, I...) = arg.blocks[I...]

# Specialize arguments to variables and parameters
const VariableArgument = Argument{AtomicVariable}
const VariableArgumentBlocks = ArgumentBlocks{AtomicVariable}
const ConstantArgument = Argument{AtomicConstant}
const ConstantArgumentBlocks = ArgumentBlocks{AtomicConstant}

"""
`AffineFunction` defines an affine function that is used to create conic
constraints in geometric form. The function accepts a list of argument blocks
(variables and parameters). During creation time, the user specifies the value
and Jacobians of the function. In particular, the following can be evaluated:

- `f(x, p)` (via the `f` member). This returns an `n`-element
  `Vector{Float64}`.
- `∇x f(x, p)` (via the `Jx` member). This returns an `n,n`-element
  `Matrix{Float64}`.
- `∇p f(x, p)` (via the `Jp` member). This returns an `n,n`-element
  `Matrix{Float64}`.
- `∇px f(x, p)` (via the `Jpx` member). This returns an `n,n,d`-element
  `Array{Float64, 3}` tensor. Each index along the third dimension represents a
  Hessian with respect to a single parameter, i.e. a derivative of `∇x f` with
  respect to `p[i]`.

When evaluating the function, the values of `x` and `p` are passed and not the
wrapper `ArgumentBlock` structs themselves. So the function has to be defined
such that is is able to operate on both `AtomicVariable` and `AtomicConstant`
types.
"""
struct AffineFunction
    f::DifferentiableFunction # The core computation method
    x::VariableArgumentBlocks # Variable arguments
    p::ConstantArgumentBlocks # Constant arguments

    """
        AffineFunction(prog, x, p)

    Affine function constructor.

    # Arguments
    - `prog`: the optimization program.
    - `x`: the variable argument blocks.
    - `p`: the parameter argument blocks.
    - `f`: the core method that can compute the function value and its
      Jacobians.

    # Returns
    - `Axb`: the affine function object.
    """
    function AffineFunction(
        prog::AbstractConicProgram,
        x::VariableArgumentBlocks,
        p::ConstantArgumentBlocks,
        f::Function)::AffineFunction

        # Create the differentiable function wrapper
        xargs = length(x)
        pargs = length(p)
        consts = prog.pars[]
        f = DifferentiableFunction(f, xargs, pargs, consts)

        Axb = new(f, x, p)

        return Axb
    end # function
end # struct

"""
    Axb([args...][; jacobians])

Compute the function value and Jacobians. This basically forwards data to the
underlying `DifferentiableFunction`, which handles the computation.

# Arguments
- `args`: (optional) evaluate the function for these input argument values. If
  not provided, the function is evaluated at the values of the internally
  stored blocks on which is depends.

# Keywords
- `jacobians`: (optional) set to true in order to compute the Jacobians as
  well.

# Returns
- `f`: the function value. The Jacobians can be queried later by using the
  `jacobian` function.
"""
function (Axb::AffineFunction)(args::BlockValue{AtomicConstant}...;
                               jacobians::Bool=false)::FunctionValueType

    # Compute the input argument values
    if isempty(args)
        x_input = [value(blk) for blk in Axb.x]
        p_input = [value(blk) for blk in Axb.p]
        args = vcat(x_input, p_input)
    end

    f_value = Axb.f(args...; jacobians) # Core call

    return f_value
end # function

"""
Convenience methods that pass the calls down to `DifferentiableFunction`.
"""
value(F::AffineFunction)::FunctionValueType = value(F.f)
jacobian(F::AffineFunction,
         key::JacobianKeys)::JacobianValueType = jacobian(F.f, key)


"""
`ConicConstraint` defines a conic constraint for the optimization program. It
is basically the following mathematical object:

```math
f(x, p)\\in \\mathcal K
```

where ``f`` is an affine function and ``\\mathcal K`` is a convex cone.
"""
struct ConicConstraint
    f::AffineFunction # The affine function
    K::ConvexCone     # The convex cone set data structure
    constraint::Types.ConstraintRef # Underlying JuMP constraint
    prog::Ref{AbstractConicProgram} # The parent conic program

    """
        ConicConstraint(f, kind)

    Basic constructor. The function `f` value gets evaluated.

    # Arguments
    - `f`: the affine function which must belong to some convex cone.
    - `kind`: the kind of convex cone that is to be used. All the sets allowed
      by `ConvexCone` are supported.
    - `prog`: the parent conic program to which the constraint belongs.

    # Returns
    - `finK`: the conic constraint.
    """
    function ConicConstraint(f::AffineFunction,
                             kind::Symbol,
                             prog::AbstractConicProgram)::ConicConstraint

        # Create the underlying JuMP constraint
        f_value = f()
        K = ConvexCone(f_value, kind)
        constraint = add!(prog.mdl, K)

        finK = new(f, K, constraint, prog)

        return finK
    end # function
end # struct

""" Conic clinear program main class. """
mutable struct ConicProgram <: AbstractConicProgram
    mdl::Model          # Core JuMP optimization model
    pars::Ref           # A parameter structure used for problem definition
    x::VariableArgument # Decision variable vector
    p::ConstantArgument # Parameter vector

    """
        ConicProgram([solver][; solver_options])

    Empty model constructor.

    # Arguments
    - `pars`: problem parameter structure. This can be anything, and it is
      passed down to the low-level functions defining the problem.

    # Keywords
    - `solver`: (optional) the numerical convex optimizer to use.
    - `solver_options`: (optional) options to pass to the numerical convex
      optimizer.

    # Returns
    - `prog`: the conic linear program data structure.
    """
    function ConicProgram(
        pars::Any;
        solver::DataType=ECOS.Optimizer,
        solver_options::Union{Dict{String, Any},
                              Nothing}=nothing)::ConicProgram

        # Configure JuMP model
        mdl = Model()
        set_optimizer(mdl, solver)
        if !isnothing(solver_options)
            for (key,val) in solver_options
                set_optimizer_attribute(mdl, key, val)
            end
        end

        # Variables and parameters
        pars = Ref(pars)
        x = VariableArgument()
        p = ConstantArgument()

        # Combine everything into a conic program
        prog = new(mdl, pars, x, p)

        # Associate the arguments with the newly created program
        link!(x, prog)
        link!(p, prog)

        return prog
    end # function
end # struct

"""
    populate!(X, mdl[, sub][; name])

Populate an uninitialized array with JuMP optimization variables. This is a
recursive function.

# Arguments
- `X`: the uninitialized array.
- `mdl`: the JuMP optimization model structure.
- `sub`: (optional) subscript for element location. Used internally, do not
  provide as the user.

# Keywords
- `name`: the element base names.
"""
function populate!(X, mdl::Model, sub::String=""; name::String="")::Nothing
    if length(X)==1
        full_name = isempty(sub) ? name : name*"_"*sub
        X[1] = @variable(mdl, base_name=full_name) #noinfo
    else
        for i=1:size(X, 1)
            nsub = sub*string(i)
            populate!(view(X, i, :), mdl, nsub; name=name)
        end
    end
    return nothing
end # function

"""
    link!(arg, owner)

Link an argument to its parent program.

# Arguments
- `arg`: the argument.

# Returns
- `owner`: the program that owns this argument.
"""
function link!(arg::Argument, owner::ConicProgram)::Nothing
    arg.prog[] = owner
    return nothing
end # function

"""
    push!(arg, name, shape...)

Append a new block to the end of the argument.

# Arguments
- `arg`: the function argument to be appended to.
- `name`: the name of the new block.
- `shape...`: the new block's shape. If ommitted entirely then treated as a
  scalar.

# Returns
- `block`: the new block.
"""
function Base.push!(arg::Argument, name::String, shape::Int...)::ArgumentBlock

    # Create the new block
    blid = blocks(arg)+1
    elid1 = length(arg)+1
    block = ArgumentBlock(arg, shape, blid, elid1, name)

    # Update the arguemnt
    push!(arg.blocks, block)
    arg.numel += length(block)

    return block
end # function

"""
    push!(prog, kind, shape[; name])

Add a new argument block to the optimization program.

# Arguments
- `prog`: the optimization program.
- `kind`: the kind of argument (`:variable` or `:parameter`).
- `shape...`: the shape of the argument block.

# Returns
- `block`: the new argument block.
"""
function Base.push!(prog::ConicProgram,
                    kind::Symbol,
                    shape::Int...;
                    name::Union{String, Nothing}=nothing)::ArgumentBlock

    if !(kind in (:variable, :parameter))
        err = SCPError(0, SCP_BAD_ARGUMENT,
                       "specify either :variable or :parameter")
        throw(err)
    end

    z = (kind==:variable) ? prog.x : prog.p

    # Assign a default name if the user does not provide one
    if isnothing(name)
        base_name = (kind==:variable) ? "x" : "p"
        name = base_name*@sprintf("%d", blocks(z)+1)
    end

    block = push!(z, name, shape...)

    return block
end

""" Specialize `push!` for variables. """
function variable!(prog::ConicProgram, shape::Int...;
                   name::Union{String, Nothing}=nothing)::ArgumentBlock
    push!(prog, :variable, shape...; name=name)
end # function

""" Specialize `push!` for parameters. """
function parameter!(prog::ConicProgram, shape::Int...;
                    name::Union{String, Nothing}=nothing)::ArgumentBlock
    push!(prog, :parameter, shape...; name=name)
end # function

"""
    @new_argument(prog, shape, name, kind)

Macro to create a new variable.

# Arguments
- `prog`: the conic program object.
- `shape`: the shape of the argument, as a tuple or a vector.
- `name`: the argument name.
- `kind`: either `:variable` or `:parameter`.

# Returns
The newly created argument object.
"""
macro new_argument(prog, shape, name, kind)
    @assert typeof(kind)<:QuoteNode
    func = (kind.value==:variable) ? :(variable!) : :(parameter!)

    if typeof(shape)<:Expr
        # Array variable
        @assert shape.head == :vect || shape.head == :tuple
        @assert ndims(shape.args) == 1
        shape = shape.args
        :( $func($(esc(prog)), $(shape...); name=$name) )
    else
        # Scalar variable
        :( $func($(esc(prog)), $shape; name=$name) )
    end
end # macro

"""
    @new_variable(prog, shape, name)
    @new_variable(prog, shape)
    @new_variable(prog, name)
    @new_variable(prog)

The following macros specialize `@new_argument` for creating variables.
"""
macro new_variable(prog, shape, name)
    :( @new_argument($(esc(prog)), $shape, $name, :variable) )
end # macro

macro new_variable(prog, shape_or_name)
    if typeof(shape_or_name)<:String
        :( @new_variable($(esc(prog)), 1, $shape_or_name) )
    else
        :( @new_variable($(esc(prog)), $shape_or_name, nothing) )
    end
end # macro

macro new_variable(prog)
    :( @new_variable($(esc(prog)), 1, nothing) )
end # macro

"""
    @new_parameter(prog, shape, name)
    @new_parameter(prog, shape)
    @new_parameter(prog, name)
    @new_parameter(prog)

The following macros specialize `@new_argument` for creating parameters.
"""
macro new_parameter(prog, shape, name)
    :( @new_argument($(esc(prog)), $shape, $name, :parameter) )
end # macro

macro new_parameter(prog, shape_or_name)
    if typeof(shape_or_name)<:String
        :( @new_parameter($(esc(prog)), 1, $shape_or_name) )
    else
        :( @new_parameter($(esc(prog)), $shape_or_name, nothing) )
    end
end # macro

macro new_parameter(prog)
    :( @new_parameter($(esc(prog)), 1, nothing) )
end # macro

"""
    value(blk)

Get the value of the block.

# Arguments
- `blk`: the block.

# Returns
- `val`: the block's value.
"""
function value(blk::ArgumentBlock{T, N})::BlockValue{T, N} where {T, N}

    if T<:AtomicVariable
        mdl = blk.arg[].prog[].mdl # The underlying optimization model
        if termination_status(mdl)==MOI.OPTIMIZE_NOT_CALLED
            # Optimization not yet performed, so return the variables
            # themselves
            val = blk.value
        else
            # The variables have been assigned their optimal values, show these
            val = value.(blk.value)
        end
    else
        val = blk.value
    end

    return val
end # function
