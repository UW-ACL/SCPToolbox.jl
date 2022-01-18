#= Argument perturbation for variational problem.

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

# ..:: Globals ::..

# Constants to denote perturbationkind
@enum(PerturbationKind, FREE, FIXED, ABSOLUTE, RELATIVE)

const PerturbationKindArray = AbstractArray{PerturbationKind,N} where {N}

# ..:: Data structures ::..

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
        blk::AbstractArgumentBlock{T,N},
        kind::PerturbationKindArray,
        amount::Types.Optional{RealArray} = nothing,
    )::Perturbation{N} where {T<:AtomicArgument,N}

        if ndims(blk) > 0
            blk_shape = size(blk)
            kind = repeat(kind, outer = blk_shape .÷ size(kind))
            amount =
                isnothing(amount) ? fill(NaN, blk_shape) :
                repeat(amount, outer = blk_shape .÷ size(amount))
        elseif (length(kind) > 1 || (!isnothing(amount) && length(amount) > 1))
            msg = "kind and amount must be single-element vectors"
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        else
            kind = fill(kind[1])
            amount = isnothing(amount) ? fill(NaN) : fill(amount[1])
        end

        for i = 1:length(kind)
            if kind[i] == FREE
                amount[i] = Inf
            elseif kind[i] == FIXED
                amount[i] = isnan(amount[i]) ? 0 : amount[i]
            elseif isnan(amount[i])
                # ABSOLUTE or RELATIVE perturbation, but the perturbation
                # amount was not specified
                msg = "Perturbation is %s but amount was not specified"
                err = SCPError(0, SCP_BAD_ARGUMENT, @eval @sprintf($msg, $(kind[i])))
                throw(err)
            end
        end

        perturbation = new{N}(kind, amount)

        return perturbation
    end

    """
        Perturbation(pert, Id...)

    A "move constructor" for slicing the scaling object. By applying the same
    slicing commands as to the `ArgumentBlock`, we can recover the perturbation
    that is associated with the sliced `ArgumentBlock` object.

    # Arguments
    - `pert`: the original perturbation object.
    - `Id...`: the slicing indices/colons.

    # Returns
    - `sliced_pert`: the sliced perturbation object.
    """
    function Perturbation(pert::Perturbation, Id...)::Perturbation
        sliced_kind = view(pert.kind, Id...)
        sliced_amount = view(pert.amount, Id...)
        N = ndims(sliced_amount)
        sliced_pert = new{N}(sliced_kind, sliced_amount)
        return sliced_pert
    end
end # struct

# ..:: Methods ::..

"""
Provide an interface to make `Perturbation` behave like an array. See the
[documentation](https://docs.julialang.org/en/v1/manual/interfaces/#man-interface-array).
"""
Base.size(pert::Perturbation) = size(pert.amount)
Base.getindex(pert::Perturbation, I...) = Perturbation(pert, I...)
Base.view(pert::Perturbation, I...) = Perturbation(pert, I...)
Base.collect(pert::Perturbation) = [pert]
Base.iterate(pert::Perturbation, state::Int = 1) = iterate(pert.amount, state)

""" Get the kind of perturbation """
kind(pert::Perturbation)::PerturbationKindArray = pert.kind
amount(pert::Perturbation)::AbstractRealArray = pert.amount
