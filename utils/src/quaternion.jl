#= Quaternion type.

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
    include("helper.jl")
end

import Base: *, adjoint, vec, getindex
import ..skew

export Quaternion, dcm, rpy, slerp_interpolate, rotate, Log, skew

"""
`Quaternion` encodes a quaternion object. Stored in vectorized form/indexing,
use the scalar last convention. The behaviour follows the documentation in [1].

References

[1] Joan Sola, "Quaternion kinematics for the error-state Kalman filter." CoRR
2017, http://arxiv.org/abs/1711.02508.
"""
struct Quaternion
    v::RealVector # Vector part
    w::RealTypes  # Scalar part

    """
        Quaternion()

    Identity quaternion constructor.

    # Returns
    - `q`: the quaternion.
    """
    function Quaternion()::Quaternion
        q = new(zeros(3), 1.0)
        return q
    end # function

    """
        Quaternion(v, w)

    Basic constructor.

    # Arguments
    - `v`: the vector part.
    - `w`: the scalar part.

    # Returns
    - `q`: the quaternion.
    """
    function Quaternion(v::RealVector, w::RealTypes)::Quaternion
        if length(v)!=3
            err = ArgumentError("quaternion is a 4-element object.")
            throw(err)
        end

        q = new(v, w)

        return q
    end # function

    """
        Quaternion(v)

    (Pure) quaternion constructor from a vector.

    # Arguments
    - `v`: the vector part or the full quaternion in vector form.

    # Returns
    - `q`: the pure quaternion.
    """
    function Quaternion(v::RealVector)::Quaternion
        if length(v)!=3 && length(v)!=4
            msg = string("cannot construct a quaternion from ",
                         "fewer than 3 or more than 4 elements.")
            err = ArgumentError(msg)
            throw(err)
        end

        if length(v)==3
            q = Quaternion(v, 0.0)
        else
            q = Quaternion(v[1:3], v[4])
        end

        return q
    end # function

    """
        Qaternion(α, a)

    Unit quaternion from an angle-axis attitude parametrization.

    # Arguments
    - `α`: the angle (in radians).
    - `a`: the axis (internally normalized to a unit norm).

    # Returns
    - `q`: the unit quaternion.
    """
    function Quaternion(α::RealTypes, a::RealVector)::Quaternion
        if length(a)!=3
            msg = string("axis must be in R^3.")
            err = ArgumentError(msg)
            throw(err)
        end

        a /= norm(a)
        v = a*sin(α/2)
        w = cos(α/2)
        q = Quaternion(v, w)

        return q
    end # function
end # struct

"""
    q[i]

Quaternion indexing.

# Arguments
- `q`: the quaternion.
- `i`: the index.

# Returns
- `v`: the value.
"""
function getindex(q::Quaternion, i::Int)::RealTypes
    if i<0 || i>4
        err = ArgumentError("quaternion index out of bounds.")
        throw(err)
    end

    v = (i<=3) ? q.v[i] : q.w

    return v
end # function

"""
    skew(q[, side])

Skew-symmetric matrix from a quaternion.

# Arguments
- `q`: the quaternion.
- `side`: (optional) either :L or :R. In which case:
  - `:L` : q*p and let [q]*p, return the 4x4 matrix [q]
  - `:R` : q*p and let [p]*q, return the 4x4 matrix [p]

# Returns
- `S`: the skew-symmetric matrix.
"""
function skew(q::Quaternion, side::Symbol=:L)::RealMatrix
    S = RealMatrix(undef, 4, 4)
    S[1:3, 1:3] = q.w*I(3)+((side==:L) ? 1 : -1)*skew(q.v)
    S[1:3, 4] = q.v
    S[4, 1:3] = -q.v
    S[4, 4] = q.w
    return S
end # function

"""
    *(q, p)

Quaternion multiplication.

# Arguments
- `q`: the first quaternion.
- `p`: the second quaternion.

# Returns
- `r`: the resultant quaternion, r=q*p.
"""
function *(q::Quaternion, p::Quaternion)::Quaternion
    r = Quaternion(skew(q)*vec(p))
    return r
end # function

"""
    *(q, p)

Quaternion multiplication by a pure quaternion (a vector).

# Arguments
- `q`: quaternion or a 3-element vector.
- `p`: quaternion or a 3-element vector (opposity of `q`).

# Returns
- `r`: the resultant quaternion, `r=q*p`.
"""
function *(q::Union{Quaternion, RealVector},
           p::Union{Quaternion, RealVector})::Quaternion
    if typeof(q)<:Quaternion && typeof(p)<:RealVector
        if length(p)!=3
            err = ArgumentError("p must be a vector in R^3.")
            throw(err)
        end
        r = q*Quaternion(p)
    else
        if length(q)!=3
            err = ArgumentError("q must be a vector in R^3.")
            throw(err)
        end
        r = Quaternion(q)*p
    end
    return r
end # function

"""
    q'

Quaternion conjugate.

# Arguments
- `q`: the original quaternion.

# Returns
- `p`: the conjugate of the quaternion.
"""
function adjoint(q::Quaternion)::Quaternion
    p = Quaternion(-q.v, q.w)
    return p
end # function

"""
    Log(q)

Logarithmic map of a unit quaternion.

It returns the axis and angle associated with the quaternion operator.  We
assume that a unit quaternion is passed in (no checks are run to verify this).

# Arguments
- `q`: the unit quaternion.

# Returns
- `α`: the rotation angle (in radians).
- `a`: the rotation axis.
"""
function Log(q::Quaternion)::Tuple{Real, RealVector}
    nrm_qv = norm(q.v)
    α = 2*atan(nrm_qv, q.w)
    a = q.v/nrm_qv
    return α, a
end # function

"""
    rotate(v, q...)

Consider if the function is called as `rotate(v, q)`, i.e. with just two
arguments. This call rotates a vector `v` by the quaternion `q`. The rotation
can be thought of, equivalently, in two ways:

- Active rotation: `q` represents the rotation of vector `v` in the `A` frame,
  taking it from its original value to a new value `w`, which is a vector in
  the `A` frame that is rotated "away from `v`" by the amount encoded in `q`.

- Passive rotation: `q` represents a frame transformation from the `B` frame to
  the `A` frame. The vector `v` is a vector in the `B` frame, and "rotating by
  `q`" means expressing the same vector in the `A` frame.

In both cases, if you call `rotate(q', v)`, you are doing the inverse
operation. In particular, this means:

- Active rotation: rotate `v` in the "opposite" sense to `q`. If you think
  about a 2D rotation by an angle ``\\theta``, then this means rotating instead
  by an angle ``-\\theta``.

- Passive rotation: the vector `v` is a vector in the `A` frame, and "rotating
  by `q'`" means expressing the same vector in the `B` frame.

If the function is called as `rotate(v, q1, q2, ..., qN)` then it means to
perform the rotations in sequence. In effect, it expands to the following code:

```julia
w = rotate(v, qN)
...
w = rotate(w, q2)
w = rotate(w, q1)
```

The interpretation for the rotation sequence is as follows:

- Active rotation: begin with the vector `v` in the `A` frame, and rotate it by
  `q1` (according to the active rotation interpretation above), obtaining
  vector `w1`. At the same time, rotate the axes of the `A` frame by `q1`. Now
  rotate the vector `w1` (which is still an `A` frame vector!) by `q2`, except
  that `q2` specifies rotation about the axes of the new frame `A1` (this is
  called an *intrinsic rotation* [1]). This yields the new vector `w2` (again
  in the `A` frame), and a new frame `A2` which is rotated from `A1` by
  `q2`. Repeat the process until you obtain `wN`, which is the resultant final
  rotated vector. Specifically, `wN` is a vector in the `A` frame which is
  rotated from the original vector `v` by the sequence of intrinsic rotations
  `q1`, ..., `qN`.

- Passive rotation: taking the above description for the active rotation
  interpretation, the vector `v` can be interpreted as a vector in the final
  `AN` frame. The resultant vector `w` output by the `rotate` function is then
  the same vector `v` but expressed in the `A` frame. The `A` frame is related
  to `AN` by a sequence of passive rotations `q1`, ..., `qN`. In particular,
  `q1` is a passive rotation (i.e., coordinate frame change) from `A1` to `A`,
  `q2` is a passive rotation from `A2` to `A1`, ..., `qN` is a passive rotation
  from `A{N-1}` to `AN`.

Note that the above description reduces to the case of a single rotation
`rotate(v, q)` described at the start by identifying the final frame `AN` as
`B`.

References:

[1] https://en.wikipedia.org/wiki/Euler_angles#Conventions_by_intrinsic_rotations

# Arguments
- `v`: the original vector.
- `q...`: the sequence of quaternions to rotate by.

# Returns
- `w`: the final rotated vector.
"""
function rotate(v::RealVector, q::Quaternion...)::RealVector
    w = (q[end]*v*q[end]').v
    if length(q)>1
        w = rotate(w, q[1:end-1]...)
    end
    return w
end # function

"""
    dcm(q)

Compute the direction cosine matrix associated with a quaternion.

# Arguments
- `q`: the quaternion.

# Returns
- `R`: the 3x3 direction cosine matrix.
"""
function dcm(q::Quaternion)::RealMatrix
    R = (skew(q', :R)*skew(q))[1:3, 1:3]
    return R
end # function

"""
    rpy(q)

Compute Euler angle sequence associated with a quaternion.

Use the Z-Y'-X'' convention (Tait-Bryan angles [1]):
  0. Begin with the world coordinate system {X,Y,Z};
  1. First, rotate ("yaw") about Z. Obtain {X',Y',Z'};
  2. Next, rotate ("pitch") about Y'. Obtain {X'',Y'',Z''};
  2. Finally, rotate ("roll") about X''. Obtain the body coordinate system.

This returns an **active** rotation matrix R. Specifically, given a vector x in
the world coordinate system, R*x=x' which is the vector rotated by R, still
expressed in the world coordinate system. In particular, R*(e_i) gives the
principal axes of the rotated coordinate system, expressed in the world
coordinate system. If you want a passive rotation (in other words, a change of
coordinate frame for the same vector), use transpose(R).

References:

[1] https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles

# Arguments
- `q`: the quaternion.

# Returns
- `v`: a 3-tuple of the angles (yaw, pitch, roll) (in radians).
"""
function rpy(q::Quaternion)::Tuple{Real, Real, Real}
    R = dcm(q)
    pitch = acos(max(0.0, min(1.0, sqrt(R[1, 1]^2+R[2, 1]^2))))
    roll = atan(R[3, 2], R[3, 3])
    yaw = atan(R[2, 1], R[1, 1])
    return yaw, pitch, roll
end # function

"""
    vec(q)

Convert quaternion to vector form.

# Arguments
- `q`: the quaternion.

# Returns
- `q_vec`: the quaternion in vector form, scalar last.
"""
function vec(q::Quaternion)::RealVector
    q_vec = [q.v; q.w]
    return q_vec
end # function

"""
    slerp_interpolate(q0, q1, τ)

Spherical linear interpolation between two quaternions. See Section 2.7 of
[Sola2017].

References:

    @article{Sola2017,
      author = {Joan Sola},
      title = {Quaternion kinematics for the error-state Kalman filter},
      journal = {CoRR},
      year = {2017},
      url = {http://arxiv.org/abs/1711.02508}
    }

# Arguments
- `q0`: the starting quaternion.
- `q1`: the final quaternion.
- `τ`: interpolation mixing factor, in [0, 1]. When τ=0, q0 is returned; when
  τ=1, q1 is returned.

# Returns
- `qt`: the interpolated quaternion between q0 and q1.
"""
function slerp_interpolate(q0::Quaternion,
                           q1::Quaternion,
                           τ::RealTypes)::Quaternion
    τ = max(0.0, min(1.0, τ))
    Δq = q0'*q1 # Error quaternion correcting q0 to q1
    Δα, Δa = Log(Δq)
    Δq_t = Quaternion(τ*Δα, Δa)
    qt = q0*Δq_t
    return qt
end # function
