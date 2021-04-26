#= Hyperrectangle type.

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

import Base: contains

export Hyperrectangle

""" Hyperrectangle geometric object.

Hyperrectangle set `H = {x : l <= x <= u}`. """
struct Hyperrectangle
    n::Int        # Ambient space dimension
    l::RealVector # Lower bound ("lower-left" vertex)
    u::RealVector # Upper bound ("upper-right" vertex)
    # >> Scaling x=s.*y+c such that H maps to {y: -1 <= y <= 1} <<
    s::RealVector # Dilation
    c::RealVector # Offset

    """
        Hyperrectangle(l, u)

    Basic constructor.

    # Arguments
    - `l`: the lower-left vertex.
    - `u`: the upper-right vertex.

    # Returns
    - `H`: the hyperrectangle set.
    """
    function Hyperrectangle(l::RealVector, u::RealVector)::Hyperrectangle
        if length(l)!=length(u)
            err = ArgumentError("ERROR: vertex dimension mismatch.")
            throw(err)
        end
        n = length(l)
        s = (u-l)/2
        c = (u+l)/2
        H = new(n, l, u, s, c)
        return H
    end # function

    """
        Hyperrectangle((l1, u1)[, (l2, u2) ....])

    Constructor from axis ranges.

    # Arguments
    - `range`: the (min, max) range of values in the set along axis 1, 2, etc.

    # Returns
    - `H`: the hyperrectangle set.
    """
    function Hyperrectangle(
        range::Tuple{RealValue, RealValue}...)::Hyperrectangle

        n = length(range)
        l = RealVector([range[i][1] for i=1:n])
        u = RealVector([range[i][2] for i=1:n])

        H = Hyperrectangle(l, u)

        return H
    end # function

    """
        Hyperrectangle(offset, width, height, depth[, yaw, pitch, roll])

    Extrusion-like constructor for a 3D rectangular prism. Think about it like
    extruding a 3D rectangular prism in a Computer Aided Design (CAD)
    software. You create a 2D rectangle centered at c with dimensions width x
    height. You then extrude "forward" along the +z axis by the depth
    value. Afterwards, you yaw the rectangle by ±90 degrees in yaw, pitch, and
    roll.

    # Arguments
    - `offset`: the rectangular base center (aka centroid).
    - `width`: rectangular base width (along x).
    - `height`: rectangular base height (along y).
    - `depth`: extrusion depth (along +z).
    - `yaw`: (optional) the Tait-Bryan yaw angle (in degrees).
    - `pitch`: (optional) the Tait-Bryan pitch angle (in degrees).
    - `roll`: (optional) the Tait-Bryan roll angle (in degrees).

    # Returns
    - `H`: the hyperrectangle set.
    """
    function Hyperrectangle(offset::RealVector,
                              width::RealValue,
                              height::RealValue,
                              depth::RealValue;
                              yaw::RealValue=0.0,
                              pitch::RealValue=0.0,
                              roll::RealValue=0.0)::Hyperrectangle
        if yaw%90!=0 || pitch%90!=0 || roll%90!=0
            err = ArgumentError("ERROR: hyperrectangle must be axis-aligned.")
            throw(err)
        end
        # Compute the hyperrectangle min/max vertices in world frame, no offset
        l = RealVector([-width/2, -height/2, 0.0])
        u = RealVector([width/2, height/2, depth])
        # Apply rotation
        c = (angle) -> cosd(angle)
        s = (angle) -> sind(angle)
        ψ, θ, φ = yaw, pitch, roll
        Rz = RealMatrix([c(ψ) -s(ψ) 0;
                         s(ψ)  c(ψ) 0;
                         0       0  1])
        Ry = RealMatrix([c(θ) 0 s(θ);
                         0    1 0;
                         -s(θ) 0 c(θ)])
        Rx = RealMatrix([1 0     0;
                         0 c(φ) -s(φ);
                         0 s(φ)  c(φ)])
        R = Rz*Ry*Rx # **intrinsic** rotations
        lr = R*l
        ur = R*u
        l = min.(lr, ur)
        u = max.(lr, ur)
        # Apply offset
        l += offset
        u += offset
        # Save hyperrectangle
        H = Hyperrectangle(l, u)
        return H
    end # function
end # struct

"""
    contains(H, r)

Check if a point is contained in the hyperrectangle.

# Arguments
- `H`: the hyperrectangle set.
- `r`: the point.

# Returns
Boolean true iff r∈H.
"""
function contains(H::Hyperrectangle, r::RealVector)::Bool
    return all(H.l .<= r .<= H.u)
end # function
