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

export ConicProgram, blocks, value, constraints, cost, solve!
export @new_variable, @new_parameter, @add_constraint,
    @set_cost, @set_feasibility

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
            populate!(value, jump_model(arg); name=name)
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
Base.collect(blk::ArgumentBlock) = [blk]

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
`GeneralFunction` defines a general (i.e., affine or quadratic) function that
gets used as a building block in the conic program. The function accepts a list
of argument blocks (variables and parameters) that it depends on, and the
underlying `DifferentiableFunction` function that performs the value and
Jacobian computation.

By wrapping `DifferentiableFunction` in this way, the aim is to store the
variables and parameters that this otherwise generic "mathematical object" (the
function) depends on. With this information, we can automatically compute
Jacobians for the KKT conditions.
"""
struct GeneralFunction{T<:FunctionValueType}
    f::DifferentiableFunction{T} # The core computation method
    x::VariableArgumentBlocks    # Variable arguments
    p::ConstantArgumentBlocks    # Constant arguments

    """
        GeneralFunction{T}(prog, x, p)

    General (affine or quadratic) function constructor.

    # Arguments
    - `prog`: the optimization program.
    - `x`: the variable argument blocks.
    - `p`: the parameter argument blocks.
    - `f`: the core method that can compute the function value and its
      Jacobians.

    # Returns
    - `F`: the function object.
    """
    function GeneralFunction{T}(
        prog::AbstractConicProgram,
        x::VariableArgumentBlocks,
        p::ConstantArgumentBlocks,
        f::Function)::GeneralFunction{T} where {T<:FunctionValueType}

        # Create the differentiable function wrapper
        xargs = length(x)
        pargs = length(p)
        consts = prog.pars[]
        f = DifferentiableFunction{T}(f, xargs, pargs, consts)

        F = new{T}(f, x, p)

        return F
    end # function
end # struct

const AffineFunction = GeneralFunction{AffineFunctionValueType}
const QuadraticFunction = GeneralFunction{QuadraticFunctionValueType}

"""
    F([args...][; jacobians])

Compute the function value and Jacobians. This basically forwards data to the
underlying `DifferentiableFunction`, which handles the computation.

# Arguments
- `args`: (optional) evaluate the function for these input argument values. If
  not provided, the function is evaluated at the values of the internally
  stored blocks on which it depends.

# Keywords
- `jacobians`: (optional) set to true in order to compute the Jacobians as
  well.

# Returns
- `f`: the function value. The Jacobians can be queried later by using the
  `jacobian` function.
"""
function (F::GeneralFunction{T})(
    args::BlockValue{AtomicConstant}...;
    jacobians::Bool=false)::T where {T<:FunctionValueType}

    # Compute the input argument values
    if isempty(args)
        x_input = [value(blk) for blk in F.x]
        p_input = [value(blk) for blk in F.p]
        args = vcat(x_input, p_input)
    end

    f_value = F.f(args...; jacobians) # Core call

    return f_value
end # function

"""
Convenience methods that pass the calls down to `DifferentiableFunction`.
"""
value(F::GeneralFunction{T}) where T = value(F.f)
jacobian(F::GeneralFunction,
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
        constraint = add!(jump_model(prog), K)

        finK = new(f, K, constraint, prog)

        return finK
    end # function
end # struct

const Constraints = Vector{ConicConstraint}

# Get the kind of cone
kind(finK::ConicConstraint)::Symbol = kind(finK.K)

"""
`QuadraticCost` stores the objective function of the problem. The goal is to
minimize this function. The function can be at most quadratic, however for
robustness (in JuMP) it has been observed that it is best to reformulate the
problem (via epigraph form) such that this function is affine. """
struct QuadraticCost
    J::QuadraticFunction            # The core function
    jump::Types.QuadraticExpression # JuMP function object
    prog::Ref{AbstractConicProgram} # The parent conic program

    """
        QuadraticCost(J)

    Basic constructor. This will also call the appropriate JuMP function to
    associate the cost with the JuMP optimization model object.

    # Arguments
    - `J`: at most a quadratic function, that is to be minimized.
    - `prog`: the parent conic program which this is to be the cost of.

    # Returns
    - `cost`: the newly created cost function object.
    """
    function QuadraticCost(J::QuadraticFunction,
                           prog::AbstractConicProgram)::QuadraticCost

        # Create the underlying JuMP cost
        J_value = J()[1]
        set_objective_function(jump_model(prog), J_value)
        jump = objective_function(jump_model(prog))
        set_objective_sense(jump_model(prog), MOI.MIN_SENSE)

        cost = new(J, jump, prog)

        return cost
    end # function
end # struct

# Get the current objective function value
value(J::QuadraticCost) = value(J.J)[1]

""" Conic clinear program main class. """
mutable struct ConicProgram <: AbstractConicProgram
    mdl::Model                # Core JuMP optimization model
    pars::Ref                 # Problem definition parameters
    x::VariableArgument       # Decision variable vector
    p::ConstantArgument       # Parameter vector
    cost::Ref{QuadraticCost}  # The cost function to minimize
    constraints::Constraints  # List of conic constraints

    _feasibility::Bool        # Flag if feasibility problem

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

        # Constraints
        constraints = Constraints()

        # Objective to minimize (empty for now)
        cost = Ref{QuadraticCost}()
        _feasibility = true

        # Combine everything into a conic program
        prog = new(mdl, pars, x, p, cost, constraints,
                   _feasibility)

        # Associate the arguments with the newly created program
        link!(x, prog)
        link!(p, prog)

        # Add a zero objective (feasibility problem)
        cost!(prog, feasibility_cost, [], [])

        return prog
    end # function
end # struct

"""
Get the underlying JuMP optimization model object, starting from various
structures composing the conic program.
"""
jump_model(prog::ConicProgram)::Model = prog.mdl
jump_model(arg::Argument)::Model = arg.prog[].mdl
jump_model(blk::ArgumentBlock)::Model = blk.arg[].prog[].mdl

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
        full_name = isempty(sub) ? name : name*"["*sub[1:end-1]*"]"
        X[1] = @variable(mdl, base_name=full_name) #noinfo
    else
        for i=1:size(X, 1)
            nsub = sub*string(i)*","
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
    constraint!(prog, kind, f, x, p)

Create a conic constraint and add it to the problem. The heavy computation is
done by the user-supplied function `f`, which has to satisfy the requirements
of `AffineDifferentiableFunction`.

# Arguments
- `prog`: the optimization program.
- `kind`: the cone type.
- `f`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks.
- `p`: the parameter argument blocks.

# Returns
- `new_constraint`: the newly added constraint.
"""
function constraint!(prog::ConicProgram,
                     kind::Symbol,
                     f::Function,
                     x, p)::ConicConstraint
    x = VariableArgumentBlocks(collect(x))
    p = ConstantArgumentBlocks(collect(p))
    Axb = AffineFunction(prog, x, p, f)
    new_constraint = ConicConstraint(Axb, kind, prog)
    push!(prog.constraints, new_constraint)
    return new_constraint
end # function

"""
    cost!(prog, J, x, p)

Set the cost of the conic program. The heavy computation is done by the
user-supplied function `J`, which has to satisfy the requirements of
`QuadraticDifferentiableFunction`.

# Arguments
- `prog`: the optimization program.
- `J`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks.
- `p`: the parameter argument blocks.

# Returns
- `new_cost`: the newly created cost.
"""
function cost!(prog::ConicProgram,
               J::Function,
               x, p)::QuadraticCost
    x = VariableArgumentBlocks(collect(x))
    p = ConstantArgumentBlocks(collect(p))
    J = QuadraticFunction(prog, x, p, J)
    new_cost = QuadraticCost(J, prog)
    prog.cost[] = new_cost
    prog._feasibility = false
    return new_cost
end # function

"""
    new_argument(prog, shape, name, kind)

Add a new argument to the problem.

# Arguments
- `prog`: the conic program object.
- `shape`: the shape of the argument, as a tuple/vector/integer.
- `name`: the argument name.
- `kind`: either `:variable` or `:parameter`.

# Returns
The newly created argument object.
"""
function new_argument(prog::ConicProgram,
                      shape,
                      name::Union{String, Nothing},
                      kind::Symbol)::ArgumentBlock
    f = (kind==:variable) ? variable! : parameter!
    shape = collect(shape)
    return f(prog, shape...; name=name)
end # function

"""
    @new_variable(prog, shape, name)
    @new_variable(prog, shape)
    @new_variable(prog, name)
    @new_variable(prog)

The following macros specialize `@new_argument` for creating variables.
"""
macro new_variable(prog, shape, name)
    var = QuoteNode(:variable)
    :( new_argument($(esc.([prog, shape, name, var])...)) )
end # macro

macro new_variable(prog, shape_or_name)
    var = QuoteNode(:variable)
    if typeof(shape_or_name)<:String
        :( new_argument($(esc.([prog, 1, shape_or_name, var])...)) )
    else
        :( new_argument($(esc.([prog, shape_or_name, nothing, var])...)) )
    end
end # macro

macro new_variable(prog)
    :( new_argument($(esc(prog)), 1, nothing, :variable) )
end # macro

"""
    @new_parameter(prog, shape, name)
    @new_parameter(prog, shape)
    @new_parameter(prog, name)
    @new_parameter(prog)

The following macros specialize `@new_argument` for creating parameters.
"""
macro new_parameter(prog, shape, name)
    var = QuoteNode(:parameter)
    :( new_argument($(esc.([prog, shape, name, var])...)) )
end # macro

macro new_parameter(prog, shape_or_name)
    var = QuoteNode(:parameter)
    if typeof(shape_or_name)<:String
        :( new_argument($(esc.([prog, 1, shape_or_name, var])...)) )
    else
        :( new_argument($(esc.([prog, shape_or_name, nothing, var])...)) )
    end
end # macro

macro new_parameter(prog)
    :( new_argument($(esc(prog)), 1, nothing, :parameter) )
end # macro

"""
    @add_constraint(prog, kind, f, x, p)
    @add_constraint(prog, kind, f, x)

Add a conic constraint to the optimization problem. This is just a wrapper of
the function `constraint!`, so look there for more info.

# Arguments
- `prog`: the optimization program.
- `kind`: the cone type.
- `f`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks, as a vector/tuple/single element.
- `p`: (optional) the parameter argument blocks, as a vector/tuple/single
  element.

# Returns
The newly created `ConicConstraint` object.
"""
macro add_constraint(prog, kind, f, x, p)
    :( constraint!($(esc.([prog, kind, f, x, p])...)) )
end # macro

macro add_constraint(prog, kind, f, x)
    :( constraint!($(esc.([prog, kind, f, x, []])...)) )
end # macro

"""
    @set_cost(prog, J, x, p)
    @set_cost(prog, J, x)
    @set_feasibility(prog, J)

Set the optimization problem cost. This is just a wrapper of the function
`cost!`, so look there for more info. When both `x` and `p` arguments are
ommitted, the cost is constant so this must be a feasibility problem. In this
case, the macro name is `@set_feasibility`

# Arguments
- `prog`: the optimization program.
- `J`: the core method that can compute the function value and its Jacobians.
- `x`: (optional) the variable argument blocks, as a vector/tuple/single
  element.
- `p`: (optional) the parameter argument blocks, as a vector/tuple/single
  element.

# Returns
- `bar`: description.
"""
macro set_cost(prog, J, x, p)
    :( cost!($(esc.([prog, J, x, p])...)) )
end # macro

macro set_cost(prog, J, x)
    :( cost!($(esc.([prog, J, x, []])...)) )
end # macro

macro set_feasibility(prog)
    quote
        $(esc(prog))._feasibility = true
        cost!($(esc(prog)), feasibility_cost, [], [])
    end
end # macro

is_feasibility(prog::ConicProgram)::Bool = prog._feasibility

"""
    feasibility_cost(pars, jacobians)

A dummy function which represents a feasibility cost.

# Arguments
- `args...`: all arguments are ignored.

# Returns
- `J`: a constant-zero function representing the cost of a feasibility
  optimization problem.
"""
function feasibility_cost(
    args...)::DifferentiableFunctionOutput #nowarn

    J = DifferentiableFunctionOutput(0.0)
    return J
end # function

"""
    value(blk)

Get the value of the block.

# Arguments
- `blk`: the block.

# Returns
- `val`: the block's value.
"""
function value(blk::ArgumentBlock{T, N})::AbstractArray where {T, N}

    if T<:AtomicVariable
        mdl = jump_model(blk) # The underlying optimization model
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
    constraints(prg[, i])

Get all or the `i`th constraint.

# Arguments
- `prg`: the optimization problem.
- `i`: (optional) which constraint to get.

# Returns
The constraint(s).
"""
function constraints(prg::ConicProgram,
                     i::Int=-1)::Union{Constraints, ConicConstraint}
    if i>=0
        return prg.constraints[i]
    else
        return prg.constraints
    end
end # function

""" Get the optimization problem cost. """
cost(prg::ConicProgram)::QuadraticCost = prg.cost[]

"""
    sovle!

Description.

# Arguments
- `foo`: description.


# Returns
- `bar`: description.
"""
function solve!(prg::ConicProgram)
    mdl = jump_model(prg)
    optimize!(mdl)
    status = termination_status(mdl)
    # TODO set the dual variables
    return status
end # function
