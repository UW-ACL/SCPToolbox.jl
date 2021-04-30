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
    using .Utils
end

using JuMP
using ECOS

export ConicProgram, blocks, variable!, value

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
  well. TODO

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

    f_value = Axb.f(args...) # Core call

    return f_value
end # function

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

"""
    print_indices(id[, limit])

Print an vector of array linear indices, with a limit on how many to print in
total.

# Arguments
- `id`: the index vector.
- `limit`: (optional) half the maximum number of indices to show (at start and
  end).

# Returns
- `ids`: the index vector in string format, ready to print.
"""
function print_indices(id::LocationIndices, limit::Int=3)::String

    # Check if a single number
    if ndims(id)==0 || length(id)==1
        ids = @sprintf("%d", id[1])
        return ids
    end

    # Check if contiguous range
    iscontig = !any(diff(id).!=1)
    if iscontig
        ids = @sprintf("%d:%d", id[1], id[end])
        return ids
    end

    # Print a limited number of indices
    if length(id)>2*limit
        v0 = (@sprintf("%s", id[1:limit]))[2:end-1]
        vf = (@sprintf("%s", id[end-limit+1:end]))[2:end-1]
        ids = @sprintf("%s, ..., %s", v0, vf)
    else
        ids = @sprintf("%s", id)[2:end-1]
    end
    return ids
end # function

"""
    show(io, arg)

Pretty print the atomic (block) function argument or a slice of it.

# Arguments
- `arg`: the argument.
"""
function Base.show(io::IO, arg::ArgumentBlock{T})::Nothing where T
    compact = get(io, :compact, false) #noinfo

    isvar = T==AtomicVariable
    dim = ndims(arg)
    qualifier = Dict(0=>"Scalar", 1=>"Vector", 2=>"Matrix", 3=>"Tensor")
    if dim<=3
        qualifier = qualifier[dim]
    else
        qualifier = "N-dimensional"
    end

    kind = isvar ? "variable" : "parameter"

    @printf(io, "%s %s\n", qualifier, kind)
    @printf(io, "  %d elements\n", length(arg))
    @printf(io, "  %s shape\n", size(arg))
    @printf(io, "  Name: %s\n", name)
    @printf(io, "  Block index in stack: %d\n", arg.blid)
    @printf(io, "  Indices in stack: %s\n", print_indices(arg.elid))
    @printf(io, "  Type: %s\n", typeof(arg.value))
    if typeof(arg.value)<:AbstractArray
        @printf(io, "  Value:\n")
        io_value = IOBuffer()
        io2_value = IOContext(io_value, :limit=>true, :displaysize=>(10, 50))
        show(io2_value, MIME("text/plain"), arg.value)
        value_str = String(take!(io_value))
        # Print line by line
        for row in split(value_str, "\n")[2:end]
            @printf(io, "   %s\n", row)
        end
    else
        @printf(io, "  Value: %s\n", arg.value)
    end

    return nothing
end # function

Base.display(arg::ArgumentBlock) = show(stdout, arg)

"""
    show(io, arg)

Pretty print an argument.

# Arguments
- `arg`: argument structure.
"""
function Base.show(io::IO, arg::Argument{T})::Nothing where {T<:AtomicArgument}
    compact = get(io, :compact, false) #noinfo

    isvar = T<:AtomicVariable
    kind = isvar ? "Variable" : "Parameter"
    n_blocks = blocks(arg)
    indent = " "^get(io, :indent, 0)

    @printf(io, "%s%s argument\n", indent, kind)
    @printf(io, "%s  %d elements\n", indent, length(arg))
    @printf(io, "%s  %d blocks\n", indent, n_blocks)

    if n_blocks==0
        return nothing
    end

    ids = (i) -> arg.blocks[i].elid
    make_span_str = (i) -> print_indices(ids(i))
    span_str = [make_span_str(i) for i=1:n_blocks]
    max_span_sz = maximum(length.(span_str))
    for i = 1:n_blocks
        newline = (i==n_blocks) ? "" : "\n"
        span_str = make_span_str(i)
        span_diff = max_span_sz-length(span_str)
        span_str = span_str*(" "^span_diff)
        @printf(io, "%s   %d) %s ... %s%s",
                indent, i, span_str, arg.blocks[i].name, newline)
    end

    return nothing
end # function

"""
    show(io, prog)

Pretty print the conic program.

# Arguments
- `prog`: the conic program data structure.
"""
function Base.show(io::IO, prog::ConicProgram)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf(io, "Conic linear program\n")
    @printf(io, "  %d variables (%d blocks)\n", length(prog.x), blocks(prog.x))
    @printf(io, "  %d parameters (%d blocks)", length(prog.p), blocks(prog.p))

    if !compact
        io2 = IOContext(io, :indent=>2)
        @printf("\n\n")
        show(io2, prog.x)
        @printf("\n\n")
        show(io2, prog.p)
    end

    return nothing
end # function
