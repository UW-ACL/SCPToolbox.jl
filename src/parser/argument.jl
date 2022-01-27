#= Argument object of a conic optimization problem.

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

export VariableArgumentBlock, ConstantArgumentBlock
export value, name

# ..:: Globals ::..

# Specialize argument blocks to variables and parameters
const VariableArgumentBlock = ArgumentBlock{AtomicVariable}
const ConstantArgumentBlock = ArgumentBlock{AtomicConstant}
const ArgumentBlocks = Vector{ArgumentBlock{T}} where {T}

# ..:: Data structures ::..

"""
`Argument` stores a complete argument vector (variable or parameter) of the
optimization problem. You can think of it as being created by stacking
`ArgumentBlock` objects.
 """
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
    end
end

# ..:: Methods ::..

""" Get the total number of atomic arguments. """
numel(arg::Argument) = arg.numel

""" Support some array operations for Argument. """
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
function Base.iterate(
    arg::Argument,
    state::Int = 1,
)::Union{Nothing,Tuple{ArgumentBlock,Int}}
    if state > length(arg)
        return nothing
    else
        return arg.blocks[state], state + 1
    end
end

""" Get the underlying JuMP optimization model object """
jump_model(arg::Argument)::Model = arg.prog[].mdl

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
    blid = length(arg) + 1
    elid1 = numel(arg) + 1
    block = ArgumentBlock(arg, shape, blid, elid1, name)

    # Update the arguemnt
    push!(arg.blocks, block)
    arg.numel += length(block)

    return block
end

"""
    link!(arg, owner)

Link an argument to its parent program.

# Arguments
- `arg`: the argument.

# Returns
- `owner`: the program that owns this argument.
"""
function link!(arg::Argument, owner::AbstractConicProgram)::Nothing
    arg.prog[] = owner
    return nothing
end
