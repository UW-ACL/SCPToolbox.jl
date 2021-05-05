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

import JuMP: value, dual

export ConicProgram, numel, value, dual, constraints, name, cost, solve!,
    vary!
export @new_variable, @new_parameter, @add_constraint,
    @set_cost, @set_feasibility

const AtomicVariable = VariableRef
const AtomicConstant = Float64
const AtomicArgument = Union{AtomicVariable, AtomicConstant}
const BlockValue{T,N} = AbstractArray{T,N}
const LocationIndices = Types.IntVector

# Symbols denoting a variable or a parameter
const VARIABLE = :variable
const PARAMETER = :parameter

abstract type AbstractArgument{T<:AtomicArgument} end
abstract type AbstractConicProgram end

""" A block in the overall argument. """
struct ArgumentBlock{T<:AtomicArgument, N} <: AbstractArray{T, N}
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
Base.iterate(blk::ArgumentBlock, state::Int=1) = iterate(blk.value, state)

const BlockBroadcastStyle = Broadcast.ArrayStyle{ArgumentBlock}
Broadcast.BroadcastStyle(::Type{<:ArgumentBlock}) = BlockBroadcastStyle()
Broadcast.broadcastable(blk::ArgumentBlock) = blk

""" Get the block name. """
name(blk::ArgumentBlock)::String = blk.name

""" Get the kind of block (`VARIABLE` or `PARAMETER`). """
function kind(::ArgumentBlock{T})::Symbol where {T<:AtomicArgument}
    if T<:AtomicVariable
        return VARIABLE
    else
        return PARAMETER
    end
end # function

""" Get the indices in the argument block. """
slice_indices(blk::ArgumentBlock)::LocationIndices = blk.elid

""" Get the block index in the argument. """
block_index(blk::ArgumentBlock)::Int = blk.blid

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

""" Get the total number of atomic arguments. """
numel(arg::Argument) = arg.numel

"""
Support some array operations for Argument.
"""
Base.length(arg::Argument) = length(arg.blocks)
Base.getindex(arg::Argument, I...) = arg.blocks[I...]

"""
    iterate(arg[, state])

Iterate over the blocks of an argument.

# Arguments
- `arg`: the argument object.
- `state`: (internal) the current iterator state.

# Returns
The current block and the next state, or `nothing` if at the end.
"""
function Base.iterate(arg::Argument,
                      state::Int=1)::Union{
                          Nothing, Tuple{ArgumentBlock, Int}}
    if state > length(arg)
        return nothing
    else
        return arg.blocks[state], state+1
    end
end # function

# Specialize arguments to variables and parameters
const VariableArgument = Argument{AtomicVariable}
const VariableArgumentBlocks = ArgumentBlocks{AtomicVariable}
const ConstantArgument = Argument{AtomicConstant}
const ConstantArgumentBlocks = ArgumentBlocks{AtomicConstant}

"""
`ProgramFunction` defines an affine or quadratic function that gets used as a
building block in the conic program. The function accepts a list of argument
blocks (variables and parameters) that it depends on, and the underlying
`DifferentiableFunction` function that performs the value and Jacobian
computation.

By wrapping `DifferentiableFunction` in this way, the aim is to store the
variables and parameters that this otherwise generic "mathematical object" (the
function) depends on. With this information, we can automatically compute
Jacobians for the KKT conditions.
"""
struct ProgramFunction
    f::DifferentiableFunction # The core computation method
    x::VariableArgumentBlocks # Variable arguments
    p::ConstantArgumentBlocks # Constant arguments

    """
        ProgramFunction(prog, x, p)

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
    function ProgramFunction(
        prog::AbstractConicProgram,
        x::VariableArgumentBlocks,
        p::ConstantArgumentBlocks,
        f::Function)::ProgramFunction

        # Create the differentiable function wrapper
        xargs = length(x)
        pargs = length(p)
        consts = prog.pars[]
        f = DifferentiableFunction(f, xargs, pargs, consts)

        F = new(f, x, p)

        return F
    end # function
end # struct

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
function (F::ProgramFunction)(
    args::BlockValue{AtomicConstant}...;
    jacobians::Bool=false)::FunctionValueType

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
value(F::ProgramFunction) = value(F.f)
jacobian(F::ProgramFunction,
         key::JacobianKeys)::JacobianValueType = jacobian(F.f, key)

""" Convenience getters. """
variables(F::ProgramFunction)::VariableArgumentBlocks = F.x
parameters(F::ProgramFunction)::ConstantArgumentBlocks = F.p

"""
`ConicConstraint` defines a conic constraint for the optimization program. It
is basically the following mathematical object:

```math
f(x, p)\\in \\mathcal K
```

where ``f`` is an affine function and ``\\mathcal K`` is a convex cone.
"""
struct ConicConstraint
    f::ProgramFunction              # The affine function
    K::ConvexCone                   # The convex cone set data structure
    constraint::ConstraintRef       # Underlying JuMP constraint
    prog::Ref{AbstractConicProgram} # The parent conic program
    name::String                    # A name for the constraint

    """
        ConicConstraint(f, kind, prog[; refname, dual])

    Basic constructor. The function `f` value gets evaluated.

    # Arguments
    - `f`: the affine function which must belong to some convex cone.
    - `kind`: the kind of convex cone that is to be used. All the sets allowed
      by `ConvexCone` are supported.
    - `prog`: the parent conic program to which the constraint belongs.

    # Keywords
    - `refname`: (optional) a name for the constraint, which can be used to more
      easily search for it in the constraints list.
    - `dual`: (optional) if true, then constrain f to lie inside the dual cone.

    # Returns
    - `finK`: the conic constraint.
    """
    function ConicConstraint(f::ProgramFunction,
                             kind::Symbol,
                             prog::AbstractConicProgram;
                             refname::Union{String, Nothing}=nothing,
                             dual::Bool=false)::ConicConstraint

        # Create the underlying JuMP constraint
        f_value = f()
        K = ConvexCone(f_value, kind; dual=dual)
        constraint = add!(jump_model(prog), K)

        # Assign the name
        if isnothing(refname)
            # Create a default name
            constraint_count = length(constraints(prog))
            refname = @sprintf("f%d", constraint_count+1)
        else
            # Deconflict duplicate name by appending a number suffix
            all_names = [name(C) for C in constraints(prog)]
            matches = length(findall(
                (this_name)->occursin(this_name, refname), all_names))
            if matches>0
                refname = @sprintf("%s%d", refname, matches)
            end
        end

        finK = new(f, K, constraint, prog, refname)

        return finK
    end # function
end # struct

const Constraints = Vector{ConicConstraint}

""" Get the cone. """
cone(C::ConicConstraint)::ConvexCone = C.K

""" Get the kind of cone. """
kind(C::ConicConstraint)::Symbol = kind(cone(C))

""" Get the cone name. """
name(C::ConicConstraint)::String = C.name

""" Get the cone dual variable. """
dual(C::ConicConstraint)::Types.RealVector = dual(C.constraint)

""" Get the underlying affine function. """
lhs(C::ConicConstraint)::ProgramFunction = C.f

"""
`QuadraticCost` stores the objective function of the problem. The goal is to
minimize this function. The function can be at most quadratic, however for
robustness (in JuMP) it has been observed that it is best to reformulate the
problem (via epigraph form) such that this function is affine. """
struct QuadraticCost
    J::ProgramFunction              # The core function
    jump::Types.Variable            # JuMP function object
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
    function QuadraticCost(J::ProgramFunction,
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

""" Get the actual underlying cost function. """
core_function(J::QuadraticCost)::ProgramFunction = J.J

""" Get the current objective function value. """
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
        ConicProgram([pars][; solver, solver_options])

    Empty model constructor.

    # Arguments
    - `pars`: (optional )problem parameter structure. This can be anything, and
      it is passed down to the low-level functions defining the problem.

    # Keywords
    - `solver`: (optional) the numerical convex optimizer to use.
    - `solver_options`: (optional) options to pass to the numerical convex
      optimizer.

    # Returns
    - `prog`: the conic linear program data structure.
    """
    function ConicProgram(
        pars::Any=nothing;
        solver::DataType=ECOS.Optimizer,
        solver_options::Union{Dict{String}, Nothing}=nothing)::ConicProgram

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
    blid = length(arg)+1
    elid1 = numel(arg)+1
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
- `kind`: the kind of argument (`VARIABLE` or `PARAMETER`).
- `shape...`: the shape of the argument block.

# Returns
- `block`: the new argument block.
"""
function Base.push!(prog::ConicProgram,
                    kind::Symbol,
                    shape::Int...;
                    blk_name::Union{String, Nothing}=nothing)::ArgumentBlock

    if !(kind in (VARIABLE, PARAMETER))
        err = SCPError(0, SCP_BAD_ARGUMENT,
                       "specify either VARIABLE or PARAMETER")
        throw(err)
    end

    z = (kind==VARIABLE) ? prog.x : prog.p

    # Assign the name
    if isnothing(blk_name)
        # Create a default name
        base_name = (kind==VARIABLE) ? "x" : "p"
        blk_name = base_name*@sprintf("%d", length(z)+1)
    else
        # Deconflict duplicate name by appending a number suffix
        all_names = [name(blk) for blk in z]
        matches = length(findall(
            (this_name)->occursin(this_name, blk_name), all_names))
        if matches>0
            blk_name = @sprintf("%s%d", blk_name, matches)
        end
    end

    block = push!(z, blk_name, shape...)

    return block
end

""" Specialize `push!` for variables. """
function variable!(prog::ConicProgram, shape::Int...;
                   name::Union{String, Nothing}=nothing)::ArgumentBlock
    push!(prog, VARIABLE, shape...; blk_name=name)
end # function

""" Specialize `push!` for parameters. """
function parameter!(prog::ConicProgram, shape::Int...;
                    name::Union{String, Nothing}=nothing)::ArgumentBlock
    push!(prog, PARAMETER, shape...; blk_name=name)
end # function

"""
    constraint!(prog, kind, f, x, p)

Create a conic constraint and add it to the problem. The heavy computation is
done by the user-supplied function `f`, which has to satisfy the requirements
of `DifferentiableFunction`.

# Arguments
- `prog`: the optimization program.
- `kind`: the cone type.
- `f`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks.
- `p`: the parameter argument blocks.

# Keywords
- `name`: (optional) a name for the constraint, which can be used to more
  easily search for it in the constraints list.
- `dual`: (optional) if true, then constrain f to lie inside the dual cone.

# Returns
- `new_constraint`: the newly added constraint.
"""
function constraint!(prog::ConicProgram,
                     kind::Symbol,
                     f::Function,
                     x, p;
                     name::Union{String, Nothing}=nothing,
                     dual::Bool=false)::ConicConstraint
    x = VariableArgumentBlocks(collect(x))
    p = ConstantArgumentBlocks(collect(p))
    Axb = ProgramFunction(prog, x, p, f)
    new_constraint = ConicConstraint(Axb, kind, prog; refname=name, dual=dual)
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
    J = ProgramFunction(prog, x, p, J)
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
- `kind`: either `VARIABLE` or `PARAMETER`.

# Returns
The newly created argument object.
"""
function new_argument(prog::ConicProgram,
                      shape,
                      name::Union{String, Nothing},
                      kind::Symbol)::ArgumentBlock
    f = (kind==VARIABLE) ? variable! : parameter!
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
    var = QuoteNode(VARIABLE)
    :( new_argument($(esc.([prog, shape, name, var])...)) )
end # macro

macro new_variable(prog, shape_or_name)
    var = QuoteNode(VARIABLE)
    if typeof(shape_or_name)<:String
        :( new_argument($(esc.([prog, 1, shape_or_name, var])...)) )
    else
        :( new_argument($(esc.([prog, shape_or_name, nothing, var])...)) )
    end
end # macro

macro new_variable(prog)
    :( new_argument($(esc(prog)), 1, nothing, VARIABLE) )
end # macro

"""
    @new_parameter(prog, shape, name)
    @new_parameter(prog, shape)
    @new_parameter(prog, name)
    @new_parameter(prog)

The following macros specialize `@new_argument` for creating parameters.
"""
macro new_parameter(prog, shape, name)
    var = QuoteNode(PARAMETER)
    :( new_argument($(esc.([prog, shape, name, var])...)) )
end # macro

macro new_parameter(prog, shape_or_name)
    var = QuoteNode(PARAMETER)
    if typeof(shape_or_name)<:String
        :( new_argument($(esc.([prog, 1, shape_or_name, var])...)) )
    else
        :( new_argument($(esc.([prog, shape_or_name, nothing, var])...)) )
    end
end # macro

macro new_parameter(prog)
    :( new_argument($(esc(prog)), 1, nothing, PARAMETER) )
end # macro

"""
    @add_constraint(prog, kind, name, f, x, p)
    @add_constraint(prog, kind, name, f, x)
    @add_constraint(prog, kind, f, x, p)
    @add_constraint(prog, kind, f, x)

Add a conic constraint to the optimization problem. This is just a wrapper of
the function `constraint!`, so look there for more info.

# Arguments
- `prog`: the optimization program.
- `kind`: the cone type.
- `name`: (optional) the constraint name.
- `f`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks, as a vector/tuple/single element.
- `p`: (optional) the parameter argument blocks, as a vector/tuple/single
  element.

# Returns
The newly created `ConicConstraint` object.
"""
macro add_constraint(prog, kind, name, f, x, p)
    :( constraint!($(esc.([prog, kind, f, x, p])...);
                   name=$name) )
end # macro

macro add_constraint(prog, kind, name_f, f_x, x_p)
    if typeof(name_f)<:String
        :( constraint!($(esc.([prog, kind, f_x, x_p, []])...);
                       name=$name_f) )
    else
        :( constraint!($(esc.([prog, kind, name_f, f_x, x_p])...)) )
    end
end # macro

macro add_constraint(prog, kind, f, x)
    :( constraint!($(esc.([prog, kind, f, x, []])...)) )
end # macro

"""
    @add_dual_constraint(prog, kind, name, f, x, p)
    @add_dual_constraint(prog, kind, name, f, x)
    @add_dual_constraint(prog, kind, f, x, p)
    @add_dual_constraint(prog, kind, f, x)

These macros work exactly like `@add_constraint`, except the value `f` is
imposed to lie inside the dual of the cone `kind`.
"""
macro add_dual_constraint(prog, kind, name, f, x, p)
    :( constraint!($(esc.([prog, kind, f, x, p])...);
                   name=$name, dual=true) )
end # macro

macro add_dual_constraint(prog, kind, name_f, f_x, x_p)
    if typeof(name_f)<:String
        :( constraint!($(esc.([prog, kind, f_x, x_p, []])...);
                       name=$name_f, dual=true) )
    else
        :( constraint!($(esc.([prog, kind, name_f, f_x, x_p])...)) )
    end
end # macro

macro add_dual_constraint(prog, kind, f, x)
    :( constraint!($(esc.([prog, kind, f, x, []])...);
                   dual=true) )
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
    constraints(prg[, ref])

Get all or some of the constraints. If `ref` is a number or slice, then get
constraints from the list by regular array slicing operation. If `ref` is a
string, return all the constraints whose name contains the string `ref`.

# Arguments
- `prg`: the optimization problem.
- `ref`: (optional) which constraint to get.

# Returns
The constraint(s).
"""
function constraints(prg::ConicProgram,
                     ref=-1)::Union{Constraints, ConicConstraint}
    if typeof(ref)<:String
        # Search for all constraints the match `ref`
        match_list = Vector{ConicConstraint}(undef, 0)
        for constraint in prg.constraints
            if occursin(ref, name(constraint))
                push!(match_list, constraint)
            end
        end
        return match_list
    else
        # Get the constraint by numerical reference
        if ref>=0
            return prg.constraints[ref]
        else
            return prg.constraints
        end
    end
end # function

""" Get the optimization problem cost. """
cost(prg::ConicProgram)::QuadraticCost = prg.cost[]

"""
    solve!(prg)

Solve the optimization problem.

# Arguments
- `prg`: the optimization problem structure.

# Returns
- `status`: the termination status code.
"""
function solve!(prg::ConicProgram)::MOI.TerminationStatusCode
    mdl = jump_model(prg)
    optimize!(mdl)
    status = termination_status(mdl)
    return status
end # function

"""
    hash(blk)

Hash function for the argument block.

# Arguments
- `blk`: the argument block.

# Returns
Hash number.
"""
function Base.hash(blk::ArgumentBlock)::UInt64
    return hash(blk.name)
end # function

"""
    copy(blk, prg)

Copy the argument block (in aspects like shape and name) to a different
optimization problem. The newly created argument gets inserted at the end of
the corresponding argument of the new problem

# Arguments
- `foo`: description.
- `prg`: the destination optimization problem for the new variable.

# Keywords
- `new_name`: (optional) a format string for creating the new name from the
  original block's name.
- `copyas`: (optional) force copy the block as a `VARIABLE` or `PARAMETER`.

# Returns
- `new_blk`: the newly created argument block.
"""
function Base.copy(blk::ArgumentBlock{T},
                   prg::ConicProgram;
                   new_name::String="%s",
                   copyas::Union{Symbol,
                                 Nothing}=nothing)::ArgumentBlock where T

    blk_shape = size(blk)
    blk_kind = isnothing(copyas) ? kind(blk) : copyas
    blk_name = name(blk)
    new_name = @eval @sprintf($new_name, $blk_name)

    new_blk = new_argument(prg, blk_shape, new_name, blk_kind)

    return new_blk
end # function

"""
    function_args_id(F, args)

Get the input argument indices of `args` for the function `F`.

# Arguments
- `F`: the function.
- `args`: the arguments of the function, a function which is either `variables`
  or `parameters`.

# Returns
The index numbers of the arguments.
"""
function function_args_id(F::ProgramFunction,
                          args::Function)::LocationIndices
    nargs = length(args(F))
    if args==variables
        return 1:nargs
    else
        return (1:nargs).+length(variables(F))
    end
end # function

"""
    fill_jacobian!(Df, args, F)

Fill the blocks of the Jacobian of a vector-valued function. Let the function
be ``f(x)``. The method fills in the blocks of ``D_x f(x)``.

# Arguments
- `Df`: a pre-initialized zero matrix to store the Jacobian.
- `args`: the arguments of the function.
- `F`: the function itself.
"""
function fill_jacobian!(Df::Types.RealMatrix,
                        args::Function,
                        F::ProgramFunction)::Nothing
    j = function_args_id(F, args)
    args = args(F)
    narg = length(args)
    for i = 1:narg
        try
            J = jacobian(F, j[i])
            # If here, the Jacobian was defined
            id_x = slice_indices(args[i])
            Df[:, id_x] = J
        catch e
            typeof(e)==SCPError || rethrow(e)
        end
    end
    return nothing
end # function

"""
    fill_jacobian!(Df, xargs, yargs, F)

Fill the blocks of the Jacobian of a matrix-valued function. Let the function
be ``f(x,y)``. This method fills in the blocks of ``D_{xy} f(x,y)``.

# Arguments
- `Df`: a pre-initialized zero matrix to store the Jacobian.
- `xargs`: the ``x``-arguments of the function.
- `yargs`: the ``y``-arguments of the function.
- `F`: the function itself.
"""
function fill_jacobian!(Df::Types.RealTensor,
                        xargs::Function,
                        yargs::Function,
                        F::ProgramFunction)::Nothing
    ki = function_args_id(F, xargs)
    kj = function_args_id(F, yargs)
    symm = xargs==yargs
    xargs, yargs = xargs(F), yargs(F)
    nargx, nargy = length(xargs), length(yargs)
    for i = 1:nargx
        for j = 1:nargy
            try
                H = jacobian(F, (ki[i], kj[j]))
                # If here, the Hessian was defined
                id_x = slice_indices(xargs[i])
                id_y = slice_indices(yargs[j])
                Df[:, id_x, id_y] = H
                if symm
                    Df[:, id_y, id_x] = H'
                end
            catch e
                typeof(e)==SCPError || rethrow(e)
            end
        end
    end
    return nothing
end # function

"""
    vary!(prg)

Compute the variation of the optimal solution with respect to changes in the
constant arguments of the problem. This sets the appropriate data such that
afterwards the `sensitivity` function can be called for each variable argument.

Internally, this function formulates the linearized KKT optimality conditions
around the optimal solution.

# Arguments
- `prg`: the optimization problem structure.

# Returns
- `bar`: description.
"""
function vary!(prg::ConicProgram)::ConicProgram

    # Initialize the variational problem
    kkt = ConicProgram()

    # Create the concatenated primal variable perturbation
    nx = numel(prg.x)
    np = numel(prg.p)
    δx = @new_variable(kkt, nx, "δx")
    δp = @new_variable(kkt, np, "δp")

    # Create the dual variable perturbations
    n_cones = length(constraints(prg))
    λ = dual.(constraints(prg))
    δλ = VariableArgumentBlocks(undef, n_cones)
    for i = 1:n_cones
        blk_name = @sprintf("δλ%d", i)
        δλ[i] = @new_variable(kkt, length(λ[i]), blk_name)
    end

    # Build the constraint function Jacobians
    f = Vector{Types.RealVector}(undef, n_cones)
    Dxf = Vector{Types.RealMatrix}(undef, n_cones)
    Dpf = Vector{Types.RealMatrix}(undef, n_cones)
    Dpxf = Vector{Types.RealTensor}(undef, n_cones)
    for i = 1:n_cones
        C = constraints(prg, i)
        F = lhs(C)
        K = cone(C)
        nf = ndims(K)

        f[i] = F(jacobians=true)

        Dxf[i] = zeros(nf, nx)
        Dpf[i] = zeros(nf, np)
        Dpxf[i] = zeros(nf, nx, np)

        fill_jacobian!(Dxf[i], variables, F)
        fill_jacobian!(Dpf[i], parameters, F)
        fill_jacobian!(Dpxf[i], parameters, variables, F)
    end

    # Build the cost function Jacobians
    J = core_function(cost(prg))
    J(jacobians=true)
    DxJ = zeros(1, nx)
    DxxJ = zeros(1, nx, nx)
    DpxJ = zeros(1, nx, np)
    fill_jacobian!(DxJ, variables, J)
    fill_jacobian!(DxxJ, variables, variables, J)
    fill_jacobian!(DpxJ, parameters, variables, J)
    DxJ = DxJ[:]
    DxxJ = DxxJ[1, :, :]
    DpxJ = DpxJ[1, :, :]

    # Primal feasibility
    for i = 1:n_cones
        C = constraints(prg, i)
        K = kind(C)
        primal_feas = (δx, δp, _, _) -> @value(f[i]+Dxf[i]*δx+Dpf[i]*δp)
        @add_constraint(kkt, K, "primal_feas", primal_feas, (δx, δp))
    end

    # Dual feasibility
    for i = 1:n_cones
        K = kind(constraints(prg, i))
        dual_feas = (δλ, _, _) -> @value(λ[i]+δλ)
        @add_dual_constraint(kkt, K, "dual_feas", dual_feas, δλ[i])
    end

    # Complementary slackness
    for i = 1:n_cones
        compl_slack = (δx, δp, δλ, _, _) -> @value(dot(f[i], δλ)+
            dot(Dxf[i]*δx+Dpf[i]*δp, λ[i]))
        @add_constraint(kkt, :zero, "compl_slack", compl_slack,
                        (δx, δp, δλ[i]))
    end

    # Stationarity
    stat = (δx, δp, args...) -> begin
        δλ = args[1:end-2]
        out = DxxJ*δx+DpxJ*δp
        Dxf_vary_p = (i) -> sum(Dpxf[i][:, :, j]*δp[j] for j=1:np)
        for i = 1:n_cones
            out -= Dxf_vary_p(i)'*λ[i]+Dxf[i]'*δλ[i]
        end
        @value(out)
    end
    @add_constraint(kkt, :zero, "stat", stat, (δx, δp, δλ...))

    return kkt
end # function
