"""
Variable scaling for the optimization problem.

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
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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

            S = isnothing(S) ? S : filldims(S, blk)
            c = isnothing(c) ? c : filldims(c, blk)

            S = isnothing(S) ? ones(size(blk)) :
                repeat(S, outer=size(blk).÷size(S))
            c = isnothing(c) ? zeros(size(blk)) :
                repeat(c, outer=size(blk).÷size(c))

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
    end

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
    end
end # struct

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

""" Simple getters for scaling terms. """
dilation(scale::Scaling)::AbstractRealArray = scale.S
offset(scale::Scaling)::AbstractRealArray = scale.c

"""
    scale(sc, x)

Apply scaling (dilation and offset) to an array.

# Arguments
- `sc`: the scaling definition object.
- `xh`: the array to be scaled.

# Returns
- `x`: the scaled array.
"""
function scale(sc::Scaling, xh::AbstractArray)::AbstractArray
    S = dilation(sc)
    c = offset(sc)
    x = (S.*xh).+c
    x = (x isa AbstractArray) ? x : fill(x)
    return x
end

"""
    filldims(A, B)

Pad A with extra dimensions so that it matches the dimension of B.

# Arguments
- `A`: the array of fewer dimensions.
- `B`: the array of more dimensions.

# Returns
- `C`: mathematically the same array as A, except stored as a higher
  dimensional array.
"""
function filldims(A::AbstractArray, B::AbstractArray)::AbstractArray
    extra_dims = ndims(B)-ndims(A)
    if extra_dims<0
        msg = "B msut be of higher dimension than A"
        err = SCPError(0, SCP_BAD_ARGUMENT, msg)
        throw(err)
    else
        C = reshape(A, (size(A)..., ones(Int, extra_dims)...))
    end
    return C
end
