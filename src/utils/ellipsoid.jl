#= Ellipsoid type.

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

export Ellipsoid, project, ∇

""" Ellipsoid geometric object.

`Ellipsoid set = {x : norm(H*(x-c), 2) <= 1},`

where H is a positive definite matrix which defines the ellipsoid shape, and c
is the ellipsoid's center. """
struct Ellipsoid
    H::RealMatrix # Ellipsoid shape matrix
    c::RealVector # Ellipsoid center

    """
        Ellipsoid(H, c)

    Basic constructor.

    # Arguments
    - `H`: ellipsoid shape matrix.
    - `c`: ellipsoid center

    # Returns
    - `E`: the ellipsoid.
    """
    function Ellipsoid(H::RealMatrix, c::RealVector)::Ellipsoid
        if size(H, 2) != length(c)
            err = ArgumentError("matrix size mismatch.")
            throw(err)
        end
        E = new(H, c)
        return E
    end
end

"""
    project(E, ax)

Project an ellipsoid onto a subset of its axes.

# Arguments
- `E`: the ellipsoid to be projected.
- `ax`: array of the axes onto which to project.

# Returns
- `E_prj`: the projected ellipsoid.
"""
function project(E::Ellipsoid, ax::IntVector)::Ellipsoid
    # Parameters
    n = length(E.c)
    m = length(ax)

    # Projection matrix onto lower-dimensional space
    P = zeros(m, n)
    for i = 1:m
        P[i, ax[i]] = 1.0
    end

    # Do the projection
    Hb = P * E.H
    F = svd(Hb)
    H_prj = F.U * diagm(F.S)
    c_prj = P * E.c
    E_prj = Ellipsoid(H_prj, c_prj)

    return E_prj
end

"""
    (E::Ellipsoid)(r)

Evaluate ellipsoid level set value at location. Acts like a functor [1].

[1] https://docs.julialang.org/en/v1/manual/methods/#Function-like-objects

# Arguments
- `r`: location at which to evaluate ellipsoid value.

# Returns
- `y`: the level set value.
"""
function (E::Ellipsoid)(r::RealVector)::Real
    y = norm(E.H * (r - E.c))
    return y
end

"""
    ∇(E, r)

Ellipsoid gradient at location.

# Arguments
- `r`: location at which to evaluate ellipsoid gradient.

# Returns
- `g`: the gradient value.
"""
function ∇(E::Ellipsoid, r::RealVector)::RealVector
    g = (E.H' * E.H) * (r - E.c) / E(r)
    return g
end
