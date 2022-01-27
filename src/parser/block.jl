#= Individual block of an argument of a conic optimization problem.

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

import JuMP: value, name

export ArgumentBlock, scale, perturbation, name

export @scale, @perturb_free, @perturb_fix, @perturb_relative, @perturb_absolute

# ..:: Globals ::..

# Constants to denote variable or a parameter
@enum(ArgumentKind, VARIABLE, PARAMETER)

const BlockValue = AbstractArray{T,N} where {T,N}

# ..:: Data structures ::..

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
struct ArgumentBlock{T<:AtomicArgument,N} <: AbstractArgumentBlock{T,N}
    value::BlockValue{T,N}       # The value of the block
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
        shape::NTuple{N,Int},
        blid::Int,
        elid1::Int,
        name::String,
    )::ArgumentBlock{T,N} where {T<:AtomicArgument,N}

        # Initialize the value
        if T <: AtomicVariable
            value = Array{AtomicVariable,N}(undef, shape...)
            populate!(value, jump_model(arg); name = name)
        else
            value = fill(NaN, shape...)
        end

        elid = (1:length(value)) .+ (elid1 - 1)
        arg_ref = Ref{AbstractArgument{T}}(arg)
        scale_ref = Ref{Scaling{N}}()
        preturb_ref = Ref{Perturbation{N}}()

        blk = new{T,N}(value, name, blid, elid, scale_ref, preturb_ref, arg_ref)

        # Apply unit scaling (i.e. "no scaling")
        scale = Scaling(blk)
        apply_scaling!(blk, scale; override = true)

        # Set a free perturbation
        set_perturbation!(blk, Inf; override = true)

        return blk
    end

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
        block::ArgumentBlock{T,N},
        Id...,
    )::ArgumentBlock{T} where {T<:AtomicArgument,N}

        # Slice the block value
        sliced_value = view(block.value, Id...)

        # Get the element indices for the slice (which are a subset of the
        # original)
        sliced_elid = block.elid[LinearIndices(block.value)[Id...]]
        if ndims(sliced_elid) < 1
            sliced_elid = fill(sliced_elid)
        end

        K = ndims(sliced_value)
        scale_ref = Ref{Scaling{K}}(scale(block)[Id...])
        perturb_ref = Ref{Perturbation{K}}(perturbation(block)[Id...])

        sliced_block = new{T,K}(
            sliced_value,
            block.name,
            block.blid,
            sliced_elid,
            scale_ref,
            perturb_ref,
            block.arg,
        )

        return sliced_block
    end
end

# ..:: Methods ::..

"""
Provide an interface to make `ArgumentBlock` behave like an array. See the
[documentation](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
"""
Base.size(blk::ArgumentBlock) = size(blk.value)
Base.getindex(blk::ArgumentBlock, I...) = ArgumentBlock(blk, I...)
Base.view(blk::ArgumentBlock, I...) = ArgumentBlock(blk, I...)
Base.setindex!(blk::ArgumentBlock{T}, v::V, i::Int) where {T,V} = blk.value[i] = v
Base.setindex!(blk::ArgumentBlock{T}, v::V, I::Vararg{Int,N}) where {T,V,N} =
    blk.value[I...] = v
Base.collect(blk::ArgumentBlock) = [blk]
Base.iterate(blk::ArgumentBlock, state::Int = 1) = iterate(blk.value, state)

const BlockBroadcastStyle = Broadcast.ArrayStyle{ArgumentBlock}
Broadcast.BroadcastStyle(::Type{<:ArgumentBlock}) = BlockBroadcastStyle()
Broadcast.broadcastable(blk::ArgumentBlock) = blk

""" Get the kind of block (`VARIABLE` or `PARAMETER`). """
function kind(::ArgumentBlock{T})::ArgumentKind where {T<:AtomicArgument}
    if T <: AtomicVariable
        return VARIABLE
    else
        return PARAMETER
    end
end

""" Simple getters. """
name(blk::ArgumentBlock)::String = blk.name
slice_indices(blk::ArgumentBlock)::LocationIndices = blk.elid
block_index(blk::ArgumentBlock)::Int = blk.blid
jump_model(blk::ArgumentBlock)::Model = blk.arg[].prog[].mdl
scale(blk::ArgumentBlock)::Scaling = blk.scale[]
perturbation(blk::ArgumentBlock)::Perturbation = blk.perturb[]

""" Wrapper to call `scale` directly with `ArgumentBlock`. """
scale(blk::ArgumentBlock, xh::AbstractArray) = scale(scale(blk), xh)

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
function apply_scaling!(
    blk::ArgumentBlock{T,N},
    scale::Scaling{N};
    override::Bool = false,
)::Nothing where {T<:AtomicArgument,N}
    if override
        blk.scale[] = scale
    else
        blk.scale[].S .= dilation(scale)
        blk.scale[].c .= offset(scale)
    end
    return nothing
end

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
function apply_perturbation!(
    blk::ArgumentBlock{T,N},
    perturb::Perturbation{N};
    override::Bool = false,
)::Nothing where {T<:AtomicArgument,N}
    if override
        blk.perturb[] = perturb
    else
        blk.perturb[].kind .= kind(perturb)
        blk.perturb[].amount .= amount(perturb)
    end
    return nothing
end

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
function set_scale!(
    blk::ArgumentBlock,
    dil::Union{Real,RealArray},
    off::Types.Optional{Union{Real,RealArray}} = nothing,
)::Nothing
    dil = (dil isa RealArray) ? dil : [dil]
    off = (isnothing(off) || off isa RealArray) ? off : [off]
    new_scaling = Scaling(blk, dil, off)
    apply_scaling!(blk, new_scaling)
    return nothing
end

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
- `kind`: (optional) the kind of perturbation.

# Keywords
- `override`: (optional) write over the existing perturbation if true. You
  shouldn't use this if you want to update an existing perturbation.
"""
function set_perturbation!(
    blk::ArgumentBlock,
    amount::Union{Real,RealArray},
    kind::Types.Optional{PerturbationKind} = nothing;
    override::Bool = false,
)::Nothing
    if amount == 0
        new_perturb = Perturbation(blk, [FIXED])
    elseif amount == Inf
        new_perturb = Perturbation(blk, [FREE])
    else
        if isnothing(kind)
            msg = "Perturbation kind must be specified"
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        end

        if kind == FREE
            msg =
                "FIXED or FREE perturbation incompatible with" *
                " non-Inf perturbation amounts"
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        end

        kind = [kind]
        amount = (amount isa RealArray) ? amount : [amount]
        new_perturb = Perturbation(blk, kind, amount)
    end

    apply_perturbation!(blk, new_perturb; override = override)

    return nothing
end

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
end

"""
    value(blk[; raw, unscaled])

Get the value of the block, optionally the raw pre-scaling value.

# Arguments
- `blk`: the block.

# Keywords
- `raw`: (optional) flag to return the variable expression and not evaluate the
  numerical value, even if the problem has been solved.
- `unscaled`: (optional) if true then return the raw underlying scaled
  value. It is to be understood that scaling does `x=S*xh+c`, and `raw=true`
  returns `xh` while `raw=false` returns `x`.

# Returns
- `val`: the block's value.
"""
function value(
    blk::ArgumentBlock{T};
    raw::Bool = false,
    unscaled::Bool = false,
)::AbstractArray where {T}

    if T <: AtomicVariable
        mdl = jump_model(blk) # The underlying optimization model
        if raw || termination_status(mdl) == MOI.OPTIMIZE_NOT_CALLED
            # Optimization not yet performed, so return the variables
            # themselves
            val = blk.value
        else
            # The variables have been assigned their optimal values, show these
            val = value.(blk.value)
            val = (val isa AbstractArray) ? val : fill(val)
        end
        # Scale the value
        val = unscaled ? val : scale(blk, val)
    else
        val = blk.value
    end

    return val
end

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
function populate!(X, mdl::Model, sub::String = ""; name::String = "")::Nothing
    if length(X) == 1
        full_name = isempty(sub) ? name : name * "[" * sub[1:end-1] * "]"
        X[1] = @variable(mdl, base_name = full_name)
    else
        for i = 1:size(X, 1)
            nsub = sub * string(i) * ","
            populate!(view(X, i, :), mdl, nsub; name = name)
        end
    end
    return nothing
end

# ..:: Macros ::..

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
    :(set_scale!($(esc.([blk, S, c])...)))
end

macro scale(blk, S)
    :(set_scale!($(esc.([blk, S, nothing])...)))
end

"""
    @perturb_free(blk)
    @perturb_fix(blk)
    @perturb_fix(blk, amount)
    @perturb_relative(blk, amount)
    @perturb_absolute(blk, amount)

Set the perturbation amount for the block. These macros just wrap
`set_perturbation!`, so look there for more info.

# Arguments
- `blk`: the argument block.
- `amount`: (optional) the amount to perturb by. Only needed for relative and
  absolute perturbatnion, and optinally can be provided to fixed perturbation
  to fix a non-zero perturbation amount.
"""
macro perturb_free(blk)
    :(set_perturbation!($(esc(blk)), Inf))
end

macro perturb_fix(blk)
    :(set_perturbation!($(esc(blk)), 0))
end

macro perturb_fix(blk, amount)
    :(set_perturbation!($(esc(blk)), $(esc(amount)), FIXED))
end

macro perturb_relative(blk, amount)
    :(set_perturbation!($(esc(blk)), $(esc(amount)), RELATIVE))
end

macro perturb_absolute(blk, amount)
    :(set_perturbation!($(esc(blk)), $(esc(amount)), ABSOLUTE))
end
