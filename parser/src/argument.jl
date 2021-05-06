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

# Constants to denote variable or a parameter
@enum(ArgumentKind, VARIABLE, PARAMETER)

# Constants to denote perturbationkind
@enum(PerturbationKind, FREE, FIXED, ABSOLUTE, RELATIVE)

const BlockValue{T,N} = AbstractArray{T,N}
const LocationIndices = Array{Int}
const RealArray{N} = Types.RealArray{N}
const AbstractRealArray{N} = AbstractArray{Float64, N}
const PerturbationKindArray{N} = AbstractArray{PerturbationKind, N}

# ..:: Data structures ::..

"""
`Scaling` holds the data used to diagonally scale an `ArgumentBlock`. As a
concrete example, if the original variable is `x`, then scaling replaces the
variable with the affine expression `x=(S.*xh).+c` where `xh` is a new "scaled"
variable and (`S`, `c`) are the dilation and offset coefficients of the same
size. Elementwise, scaling looks like:

```julia
x[i,j,...] = S[i,j,...]*xh[i,j,...]+c[i,j,...] for all i,j,...
```

The implementation uses elementwise/broadcasting operations to allow it to
naturally extend to multiple dimensions. The only limitation is that scaling
must be diagonal, i.e. 'mixing' between the elements of `xh`.
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

        if ndims(blk)>0

            blk_shape = size(blk)
            S = isnothing(S) ? ones(blk_shape) :
                repeat(S, outer=blk_shape.÷size(S))
            c = isnothing(c) ? zeros(blk_shape) :
                repeat(c, outer=blk_shape.÷size(S))

        elseif ((!isnothing(S) && length(S)>1) ||
            (!isnothing(c) && length(c)>1))

            msg = "S and c must be single-element vectors"
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        else
            S = isnothing(S) ? fill(1.0) : fill(S[1])
            c = isnothing(c) ? fill(0.0) : fill(c[1])
        end

        scaling = new{N}(S, c)
        return scaling
    end # function

    """
        Scaling(sc, Id...)

    A "move constructor" for slicing the scaling object. By applying the same
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
`Perturbation` holds information on how much to perturb an argument block by
for the variation problem in the `vary!` function.
"""
struct Perturbation{N} <: AbstractRealArray{N}
    kind::PerturbationKindArray{N} # The perturbation style
    amount::AbstractRealArray{N}   # Perturbation amount

    """
        Perturbation(kind[, amount])

    Basic constructor for the perturbation type.

    # Arguments
    - `blk`: the argument block that this perturbation corresponds to.
    - `kind`: the kind of perturbation.
    - `amount`: (optional) the amount of perturbation, which gets interpreted
      based on `mode`.

    # Returns
    - `perturb`: the perturbation object.
    """
    function Perturbation(
        blk::AbstractArgumentBlock{T, N},
        kind::PerturbationKindArray,
        amount::Types.Optional{
            RealArray}=nothing)::Perturbation{N} where { #nowarn
                T<:AtomicArgument, N}

        if ndims(blk)>0
            blk_shape = size(blk)
            kind = repeat(kind, outer=blk_shape.÷size(kind))
            amount = isnothing(amount) ? fill(NaN, blk_shape) :
                repeat(amount, outer=blk_shape.÷size(amount))
        elseif (length(kind)>1 || (!isnothing(amount) && length(amount)>1))
            msg = "kind and amount must be single-element vectors"
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        else
            kind = fill(kind[1])
            amount = isnothing(amount) ? fill(NaN) : fill(amount[1])
        end

        for i=1:length(kind)
            if kind[i]==FREE
                amount[i] = Inf
            elseif kind[i]==FIXED
                amount[i] = 0
            elseif isnan(amount[i])
                # ABSOLUTE or RELATIVE perturbation, but the perturbation
                # amount was not specified
                msg = "Perturbation is %s but amount was not specified"
                err = SCPError(
                    0, SCP_BAD_ARGUMENT, @eval @sprintf($msg, $(kind[i])))
                throw(err)
            end
        end

        perturbation = new{N}(kind, amount)

        return perturbation
    end # function

    """
        Perturbation(δ, Id...)

    A "move constructor" for slicing the scaling object. By applying the same
    slicing commands as to the `ArgumentBlock`, we can recover the perturbation
    that is associated with the sliced `ArgumentBlock` object.

    # Arguments
    - `δ`: the original perturbation object.
    - `Id...`: the slicing indices/colons.

    # Returns
    - `sliced_δ`: the sliced perturbation object.
    """
    function Perturbation(δ::Perturbation, Id...)::Perturbation
        sliced_kind = view(δ.kind, Id...)
        sliced_amount = view(δ.amount, Id...)
        N = ndims(sliced_amount)
        sliced_δ = new{N}(sliced_kind, sliced_amount)
        return sliced_δ
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
    perturb::Ref{Perturbation{N}} # Allowable perturbation
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
        preturb_ref = Ref{Perturbation{N}}()

        blk = new{T, N}(value, name, blid, elid, scale_ref,
                        preturb_ref, arg_ref)

        # Apply unit scaling (i.e. "no scaling")
        scale = Scaling(blk)
        apply_scaling!(blk, scale; override=true)

        # Set a free perturbation
        set_perturbation!(blk, Inf; override=true)

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
        perturb_ref = Ref{Perturbation{K}}(perturbation(block)[Id...])

        sliced_block = new{T, K}(sliced_value, block.name, block.blid,
                                 sliced_elid, scale_ref, perturb_ref,
                                 block.arg)

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
Provide an interface to make `Perturbation` behave like an array. See the
[documentation](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
"""
Base.size(δ::Perturbation) = size(δ.amount)
Base.getindex(δ::Perturbation, I...) = Perturbation(δ, I...)
Base.view(δ::Perturbation, I...) = Perturbation(δ, I...)
Base.collect(δ::Perturbation) = [δ]
Base.iterate(δ::Perturbation, state::Int=1) = iterate(δ.amount, state)

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

""" Get the kind of perturbation """
kind(δ::Perturbation)::PerturbationKindArray = δ.kind
amount(δ::Perturbation)::AbstractRealArray = δ.amount

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

"""
    apply_perturbation!(blk, perturb[; override])

Apply new perturbation to the argument block.

# Arguments
- `blk`: the argument block.
- `perturb`: the perturbation to apply.

# Keywords
- `override`: (optional) write over the existing perturbation if true. You
  shouldn't use this if you want to update an existing perturbation.
"""
function apply_perturbation!(blk::ArgumentBlock{T, N},
                             perturb::Perturbation{N};
                             override::Bool=false)::Nothing where {
                                 T<:AtomicArgument, N}
    if override
        blk.perturb[] = perturb
    else
        blk.perturb[].kind .= kind(perturb)
        blk.perturb[].amount .= amount(perturb)
    end
    return nothing
end # function

""" Simple getters for scaling terms. """
dilation(scale::Scaling)::AbstractRealArray = scale.S
offset(scale::Scaling)::AbstractRealArray = scale.c
scale(blk::ArgumentBlock)::Scaling = blk.scale[]

""" Simple getters for perturbation. """
amount(perturb::Perturbation)::AbstractRealArray = perturb.amount
perturbation(blk::ArgumentBlock)::Perturbation = blk.perturb[]

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
    set_perturbation!(blk, amount[, kind][; override])

Set the allowable perturbation amount for this argument block. To allow any
perturbation use `amount=Inf`. To disallow any perturbation using
`amount=0`. Otherwise, pass a scalar or vector to set the allowed perturbation
(according to array broadcasting rules) and specify if the perturbation is
absolute (`ABSOLUTE`) or relative (`RELATIVE`).

# Arguments
- `blk`: the argument block.
- `amount`: the perturbation amount.
- `kind`: the kind of perturbation (`ABSOLUTE` or `RELATIVE`).

# Keywords
- `override`: (optional) write over the existing perturbation if true. You
  shouldn't use this if you want to update an existing perturbation.
"""
function set_perturbation!(blk::ArgumentBlock,
                           amount::Union{Real, RealArray},
                           kind::Types.Optional{
                               Union{PerturbationKind,
                                     PerturbationKindArray}}=nothing;
                           override::Bool=false)::Nothing
    if amount==0
        new_perturb = Perturbation(blk, [FIXED])
    elseif amount==Inf
        new_perturb = Perturbation(blk, [FREE])
    else
        if isnothing(kind)
            err = SCPError(0, SCP_BAD_ARGUMENT,
                           "Perturbation kind must be specified")
            throw(err)
        end

        kind = (kind isa PerturbationKindArray) ? kind : [kind]

        if any(kind.==FIXED) || any(kind.==FREE)
            msg = "FIXED or FREE perturbation incompatible with non-zero"*
                " and non-Inf perturbation amounts"
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        end

        amount = (amount isa RealArray) ? amount : [amount]
        new_perturb = Perturbation(blk, kind, amount)
    end

    apply_perturbation!(blk, new_perturb; override=override)

    return nothing
end # function

"""
    @perturb_fix(blk)
    @perturb_free(blk)
    @perturb_relative(blk, amount)
    @perturb_absolute(blk, amount)

Set the perturbation amount for the block. These macros just wrap
`set_perturbation!`, so look there for more info.

# Arguments
- `blk`: the argument block.
- `amount`: (optional) the amount to perturb by. Only needed for relative and
  absolute perturbatnion.
"""
macro perturb_fix(blk)
    :( set_perturbation!($(esc(blk)), 0) )
end # macro

macro perturb_free(blk)
    :( set_perturbation!($(esc(blk)), Inf) )
end # macro

macro perturb_relative(blk, amount)
    :( set_perturbation!($(esc(blk)), $(esc(amount)), RELATIVE) )
end # macro

macro perturb_absolute(blk, amount)
    :( set_perturbation!($(esc(blk)), $(esc(amount)), ABSOLUTE) )
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
