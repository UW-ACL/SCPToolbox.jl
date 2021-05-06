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

if isdefined(@__MODULE__, :LanguageServer)
    include("general.jl")
end

import JuMP: value

export value, name
export @scale

const BlockValue{T,N} = AbstractArray{T,N}
const LocationIndices = Array{Int}
const RealArray{N} = Types.RealArray{N}
const AbstractRealArray{N} = AbstractArray{Float64, N}

# Symbols denoting a variable or a parameter
const VARIABLE = :variable
const PARAMETER = :parameter

# ..:: Data structures ::..

"""
`Scaling` holds the data used to numerically scale an `ArgumentBlock`. As a
concrete example, if the original variable is `x`, then scaling replaces the
variable with the affine expression `x=(S.*xh).+c` where `xh` is a new "scaled"
variable and (`S`, `c`) are the dilation and offset coefficients of the same
size. Elementwise, scaling looks like:

```julia
x[i,j,...] = S[i,j,...]*xh[i,j,...]+c[i,j,...]
```

The implementation uses elementwise/broadcasting operations to allow it to
naturally extend to multiple dimensions.
"""
mutable struct Scaling{N} <: AbstractRealArray{N}
    S::AbstractRealArray{N} # Dilation coefficients
    c::AbstractRealArray{N} # Scaling coefficients

    """
        Scaling(blk, S, c)

    Scaling constructor.

    # Arguments
    - `blk`: the argument block that is to be scaled.
    - `S`: (optional) the dilation coefficients.
    - `c`: (optional) the offset coefficients.

    # Returns
    - `scaling`: the resulting scaling object.
    """
    function Scaling(blk::AbstractArgumentBlock{T, N},
                     S::Types.Optional{RealArray}=nothing,
                     c::Types.Optional{RealArray}=nothing)::Scaling{N} where {
                         T<:AtomicArgument, N}
        blk_shape = size(blk)
        S = isnothing(S) ? ones(blk_shape) :
            repeat(S, outer=blk_shape.÷size(S))
        c = isnothing(c) ? zeros(blk_shape) :
            repeat(c, outer=blk_shape.÷size(S))
        scaling = new{N}(S, c)
        return scaling
    end # function

    """
        Scaling(sc, Id...)

    A "move constructor" for slicking the scaling object. By applying the same
    slicing commands as to the `ArgumentBlock`, we can recover the scaling that
    is associated with the sliced `ArgumentBlock` object.

    # Arguments
    - `sc`: the original scaling object.
    - `Id...`: the slicing indices/colons.

    # Returns
    - `sliced_scaling`: the sliced scaling object.
    """
    function Scaling(sc::Scaling, Id...)::Scaling
        sliced_S = view(sc.S, Id...)
        sliced_c = view(sc.c, Id...)
        N = ndims(sliced_S)
        sliced_scaling = new{N}(sliced_S, sliced_c)
        return sliced_scaling
    end # function
end # struct

"""
`ArgumentBlock` provides a data structure that holds a subset of the arguments
in the optimization problem. This can be any **within-block** selection of the
arguments, meaning that variables stored in `ArgumentBlock` cannot come from
two different blocks.

The chief convenience of this data structure is that it provides a direct way
to find out the indices of the particular arguments within the total argument
vector. This becomes important for creating the variation problem in the
`vary!` function.
"""
struct ArgumentBlock{T<:AtomicArgument, N} <: AbstractArgumentBlock{T, N}
    value::BlockValue{T, N}       # The value of the block
    name::String                  # Argument name
    blid::Int                     # Block number in the argument
    elid::LocationIndices         # Element indices in the argument
    scale::Ref{Scaling{N}}        # Scaling parameter
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
        name::String)::ArgumentBlock{
            T, N} where {T<:AtomicArgument, N}

        # Initialize the value
        if T<:AtomicVariable
            value = Array{AtomicVariable, N}(undef, shape...)
            populate!(value, jump_model(arg); name=name)
        else
            value = fill(NaN, shape...)
        end

        elid = (1:length(value)).+(elid1-1)
        arg_ref = Ref{AbstractArgument{T}}(arg)
        scale_ref = Ref{Scaling{N}}()

        blk = new{T, N}(value, name, blid, elid, scale_ref, arg_ref)

        # Apply unit scaling (i.e. "no scaling")
        scale = Scaling(blk)
        apply_scaling!(blk, scale; override=true)

        return blk
    end # function

    """
        ArgumentBlock(block, Id...)

    A kind of "move constructor" for slicing a block. This will **always** use
    a view into the array, so that no copy of the internal values is made. This
    means that changes the values in the resulting `ArgumentBlock` slice will
    also change the values of the original.

    # Arguments
    - `block`: the original block.
    - `Id...`: the slicing indices/colons.

    # Returns
    - `sliced_block`: the sliced block.
    """
    function ArgumentBlock(
        block::ArgumentBlock{T, N},
        Id...)::ArgumentBlock{T} where {T<:AtomicArgument, N}

        # Slice the block value
        sliced_value = view(block.value, Id...)

        # Get the element indices for the slice (which are a subset of the
        # original)
        sliced_elid = block.elid[LinearIndices(block.value)[Id...]]
        if ndims(sliced_elid)<1
            sliced_elid = fill(sliced_elid)
        end

        K = ndims(sliced_value)
        scale_ref = Ref{Scaling{K}}(scale(block)[Id...])

        sliced_block = new{T, K}(sliced_value, block.name, block.blid,
                                 sliced_elid, scale_ref, block.arg)

        return sliced_block
    end # function
end # struct

# Specialize argument blocks to variables and parameters
const VariableArgumentBlock = ArgumentBlock{AtomicVariable}
const ConstantArgumentBlock = ArgumentBlock{AtomicConstant}
const ArgumentBlocks{T} = Vector{ArgumentBlock{T}}

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
    end # function
end # struct

# Specialize arguments to variables and parameters
const VariableArgument = Argument{AtomicVariable}
const VariableArgumentBlocks = ArgumentBlocks{AtomicVariable}
const ConstantArgument = Argument{AtomicConstant}
const ConstantArgumentBlocks = ArgumentBlocks{AtomicConstant}

# ..:: Methods ::..

"""
Provide an interface to make `Scaling` behave like an array. See the
[documentation](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
"""
Base.size(sc::Scaling) = size(sc.S)
Base.getindex(sc::Scaling, I...) = Scaling(sc, I...)
Base.view(sc::Scaling, I...) = Scaling(sc, I...)
Base.collect(sc::Scaling) = [sc]
Base.iterate(sc::Scaling, state::Int=1) = iterate(sc.S, state)

"""
Provide an interface to make `ArgumentBlock` behave like an array. See the
[documentation](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
"""
Base.size(blk::ArgumentBlock) = size(blk.value)
Base.getindex(blk::ArgumentBlock, I...) = ArgumentBlock(blk, I...)
Base.view(blk::ArgumentBlock, I...) = ArgumentBlock(blk, I...)
Base.setindex!(blk::ArgumentBlock{T},
               v::V, i::Int) where {T, V} = blk.value[i] = v
Base.setindex!(blk::ArgumentBlock{T}, v::V,
               I::Vararg{Int, N}) where {T, V, N} = blk.value[I...] = v
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
function Base.iterate(arg::Argument,
                      state::Int=1)::Union{
                          Nothing, Tuple{ArgumentBlock, Int}}
    if state > length(arg)
        return nothing
    else
        return arg.blocks[state], state+1
    end
end # function

"""
Get the underlying JuMP optimization model object, starting from various
structures composing the conic program.
"""
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
    apply_scaling!(blk, scale[; override])

Apply new scaling to the argument block.

# Arguments
- `blk`: the argument block.
- `scale`: the scaling to apply.

# Keywords
- `override`: (optional) write over the existing scaling if true. You shouldn't
  use this if you want to update an existing scaling.
"""
function apply_scaling!(blk::ArgumentBlock{T, N},
                        scale::Scaling{N};
                        override::Bool=false)::Nothing where {
                            T<:AtomicArgument, N}
    if override
        blk.scale[] = scale
    else
        blk.scale[].S .= dilation(scale)
        blk.scale[].c .= offset(scale)
    end
    return nothing
end # function

""" Simple getters for scaling terms. """
dilation(scale::Scaling)::AbstractRealArray = scale.S
offset(scale::Scaling)::AbstractRealArray = scale.c
scale(blk::ArgumentBlock)::Scaling = blk.scale[]

"""
    set_scale!(blk, dil, off)

Set the scaling parameters for an argument block. For the scalar case, this
sets `x=dil*xh+off` so that `x` ends up being passed to the user functions that
define the problem, and `xh` is what the numerical optimizer works with "under
the hood".

# Arguments
- `blk`: the argument block.
- `dil`: the scaling dilation.
- `off`: the scaling offset.
"""
function set_scale!(blk::ArgumentBlock,
                    dil::Union{Real, RealArray},
                    off::Types.Optional{Union{Real,
                                              RealArray}}=nothing)::Nothing
    dil = (dil isa RealArray) ? dil : [dil]
    off = (isnothing(off) || off isa RealArray) ? off : [off]
    new_scaling = Scaling(blk, dil, off)
    apply_scaling!(blk, new_scaling)
    return nothing
end # function

"""
    @scale(blk, S, c)
    @scale(blk, S)

Scale an argument block, or a slice thereof. This is just a wrapper of the
function `set_scale!`, so look there for more info. If the internal block value
is `xh` that gets optimized by the numerical solver, then scaling will make the
user-defined functions operate on the affine-transformed value `x=(S.*xh).+c`
(not the **elementwise** multiplication and addition used in this
implementation).

# Arguments
- `blk`: the argument block.
- `S`: the scaling dilation. Can be any scalar or array, as long as it properly
  broadcasts to elemntwise multiplication.
- `c`: the scalinf osset. Can be any scalar or array, as long as it properly
  broadcasts to elemntwise edition.

# Returns
- `bar`: description.
"""
macro scale(blk, S, c)
    :( set_scale!($(esc.([blk, S, c])...)) )
end # macro

macro scale(blk, S)
    :( set_scale!($(esc.([blk, S, nothing])...)) )
end # macro

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
    scale(sc, x)

Apply scaling (dilation and offset) to an array.

# Arguments
- `sc`: the scaling definition object.
- `xh`: the array to be scaled.
"""
function scale(sc::Scaling, xh::AbstractArray)::AbstractArray
    S = dilation(sc)
    c = offset(sc)
    return (S.*xh).+c
end # function

""" Wrapper to call `scale` directly with `ArgumentBlock`. """
scale(blk::ArgumentBlock, xh::AbstractArray) = scale(scale(blk), xh)

"""
    value(blk[; raw])

Get the value of the block, optionally the raw pre-scaling value.

# Arguments
- `blk`: the block.

# Keywords
- `raw`: (optional) if true then return the raw underlying scaled value. It is
  to be understood that scaling does `x=S*xh+c`, and `raw=true` returns `xh`
  while `raw=false` returns `x`.

# Returns
- `val`: the block's value.
"""
function value(blk::ArgumentBlock{T}; raw::Bool=false)::AbstractArray where T

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
        # Scale the value
        val = raw ? val : scale(blk, val)
    else
        val = blk.value
    end

    return val
end # function
