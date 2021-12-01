#= Homotopy parameter interpolation type.

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
    include("basic_types.jl")
end

export Homotopy

""" Homotopy (continuation) object. """
struct Homotopy
    # Raw parameters
    ε::RealTypes     # Error with respect to exact 1 or 0
    δ_min::RealTypes # Most precise transition half-width
    δ_max::RealTypes # Most coarse transition half-width
    # Derived parameters
    ρ::RealTypes     # Exponential growth factor

    """
        Homotopy(δ_min[; δ_max, ε])

    Basic constructor.

    # Arguments
    - `δ_min`: the sharpest transition half-width.

    # Keywords
    - `δ_max`: (optional) the smoothest transition half-width.
    - `ε`: (optional) the y-error with respect to exact step function.

    # Returns
    - `h`: the homotopy object.
    """
    function Homotopy(
        δ_min::RealTypes; δ_max::RealTypes=1.0, ε::RealTypes=1e-2)::Homotopy

        ρ = δ_min/δ_max
        h = new(ε, δ_min, δ_max, ρ)

        return h
    end # function
end # struct

"""
    (hom::Homotopy)(x)

Interpolate the homotopy parameter value.

# Arguments
- `x`: a number between [0,1] which maps to the homotopy value. Should increase
  linearly, and the homotopy will take care of mapping to a nonlinear function.

# Returns
- `h`: homotopy value for x.
"""
function (hom::Homotopy)(x::RealTypes)::RealTypes
    h = log(1/hom.ε-1)/(hom.ρ^(x)*hom.δ_max)
    return h
end # function
