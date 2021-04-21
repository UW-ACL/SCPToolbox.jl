#= Helper functions used throughout the code.

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

using LinearAlgebra
using JuMP
using Printf
using PyPlot
using Colors

import Base: vec, adjoint, *

include("types.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public macros ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Get discrete-time trajectory values at time step k or step range {k,...,l}.

The trajectory can be represented by an array of any dimension. We assume that
the time axis is contained in the **last** dimension.

The macros below are overloaded to correspond to the following possible inputs.

# Arguments
    traj: discrete-time trajectory representation of the type Array.
    k: (optional) the time step at which to get the trajectory value. Defaults
        to the current value of k in the workspace.
    l: (optional) the final time step, in case you want to get a rank of values
        from time step k to time step l.

# Returns
    The trajectory value at time step k or range {k,...,l}.
"""
macro k(traj)
    :( $(esc(traj))[fill(:, ndims($(esc(traj)))-1)..., $(esc(:(k)))] )
end

macro k(traj, k)
    :( $(esc(traj))[fill(:, ndims($(esc(traj)))-1)..., $(esc(k))] )
end

macro k(traj, k, l)
    :( $(esc(traj))[fill(:, ndims($(esc(traj)))-1)..., $(esc(k)):$(esc(l))] )
end

""" Convenience method to get trajectory value at time k+1.

# Arguments
    traj: discrete-time trajectory representation of the type Array.

# Returns
    The trajectory value at time step k+1.
"""
macro kp1(traj)
    :( @k($(esc(traj)), $(esc(:(k)))+1) )
end

""" Convenience method to get trajectory value at time k-1.

# Arguments
    traj: discrete-time trajectory representation of the type Array.

# Returns
    The trajectory value at time step k-1.
"""
macro km1(traj)
    :( @k($(esc(traj)), $(esc(:(k)))-1) )
end

""" Convenience method to get trajectory value at the initial time.

# Arguments
    traj: discrete-time trajectory representation of the type Array.

# Returns
    The trajectory initial value.
"""
macro first(traj)
    :( @k($(esc(traj)), 1) )
end

""" Convenience method to get trajectory value at the final time.

# Arguments
    traj: discrete-time trajectory representation of the type Array.

# Returns
    The trajectory final value.
"""
macro last(traj)
    :( @k($(esc(traj)), size($(esc(traj)))[end]) )
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Add a conic constraint to the optimization problem.

# Arguments
    pbm: the optimization problem structure.
    cone: the conic constraint structure.

# Returns
    constraint: the conic constraint reference.
"""
function add_conic_constraint!(
    pbm::Model, cone::T)::T_Constraint where {T<:T_ConvexConeConstraint}

    constraint = @constraint(pbm, cone.z in cone.K)

    return constraint
end

""" Add several conic constraints to the optimization problem.

# Arguments
    pbm: the optimization problem structure.
    cones: an array of conic constraint structures.

# Returns
    constraints: the conic constraint references.
"""
function add_conic_constraints!(
    pbm::Model,
    cones::Vector{T})::T_ConstraintVector where {T<:T_ConvexConeConstraint}

    constraints = T_ConstraintVector(undef, 0)

    for cone in cones
        push!(constraints, add_conic_constraint!(pbm, cone))
    end

    return constraints
end

function fixed_cone(cone::T_ConvexConeConstraint)::T_Bool
    z_real = typeof(cone.z)<:Array{T} where {T<:Real}
    return typeof(cone.z)==T_RealVector
end

""" Generate a vector which indicates conic constraint satisfaction.

Consider the cone K which defines the constraint x∈K. Let K⊂R^n, an
n-dimensional ambient space. Let q∈R^n be an n-dimensional indicator vector,
such that q<=0 implies x∈K. Furthermore, we formulate q such that if x∈K, then
it is feasible to set q<=0. Hence, effectively, we have a bidirectional
relationship: q<=0 if and only if x∈K.

# Arguments
    pbm: the optimization problem structure.
    cone: the conic constraint structure.

# Returns
    q: the indicator vector.
"""
function get_conic_constraint_indicator!(
    pbm::Model,
    cone::T_ConvexConeConstraint)::T_OptiVar

    # Parameters
    mode = (fixed_cone(cone)) ? :numerical : :jump
    C = T_ConvexConeConstraint
    acc! = add_conic_constraint!

    # Compute the indicator
    if mode==:numerical
        z = cone.z
        if cone.kind==:zero
            q = abs.(z)
        elseif cone.kind==:nonpos
            q = z
        elseif cone.kind in (:l1, :soc, :linf)
            t = z[1]
            x = z[2:end]
            nrm = Dict(:l1 => 1, :soc => 2, :linf => Inf)
            q = norm(x, nrm[cone.kind])-t
        elseif cone.kind==:geom
            t, x = z[1], z[2:end]
            dim = cone.dim-1
            q = t-exp(1/dim*sum(log.(x)))
        elseif cone.kind==:exp
            x, y, w = z
            q = y*exp(x/y)-w
        end
    else
        z = cone.z
        if cone.kind in (:zero, :nonpos)
            q = @variable(pbm, [1:cone.dim], base_name="q")
            acc!(pbm, C(z-q, :nonpos))
            if cone.kind==:zero
                acc!(pbm, C(-q-z, :nonpos))
            end
        else
            q = @variable(pbm, base_name="q")
            if cone.kind in (:l1, :soc, :linf)
                t = z[1]
                x = z[2:end]
                acc!(pbm, C(vcat(t+q, x), cone.kind))
            elseif cone.kind==:geom
                t, x = z[1], z[2:end]
                acc!(pbm, C(vcat(x, t-q), cone.kind))
            elseif cone.kind==:exp
                x, y, w = z
                acc!(pbm, C(vcat(x, y, w+q), cone.kind))
            end
            q = [q]
        end
    end

    return q
end

"""
    get_interval(x, grid)

Compute grid bin. Get which grid interval a real number belongs to.

# Arguments
- `x`: the real number.
- `grid`: the discrete grid on the real line.

# Returns
- `k`: the interval of the grid the the real number is in.
"""
function get_interval(x::T_Real, grid::T_RealVector)::T_Int
    k = sum(x.>grid)
    if k==0
	k = 1
    end
    return k
end

""" Linear interpolation on a grid.

Linearly interpolate a discrete function on a time grid. In other words, get
the function value assuming that it is a continuous and piecewise affine
function.

# Arguments
    t: the time at which to get the function value.
    f_cps: the control points of the function, stored as columns of a matrix.
    t_grid: the discrete time nodes.

# Returns
    f_t: the function value at time t.
"""
function linterp(t::T_Real,
                 f_cps::T_RealArray,
                 t_grid::T_RealVector)::Union{T_Real,
                                              T_RealArray}
    t = max(t_grid[1], min(t_grid[end], t)) # Saturate to time grid
    k = get_interval(t, t_grid)
    c = (@kp1(t_grid)-t)/(@kp1(t_grid)-@k(t_grid))
    f_t = c*@k(f_cps) + (1-c)*@kp1(f_cps)
    return f_t
end

""" Zeroth-order hold interpolation on a grid.

Previous neighbor interpolation. The interpolated value at a query point is the
value at the previous sample grid point.

# Arguments
    t: the time at which to get the function value.
    f_cps: the control points of the function, stored as columns of a matrix.
    t_grid: the discrete time nodes.

# Returns
    f_t: the function value at time t.
"""
function zohinterp(t::T_Real,
                   f_cps::T_RealArray,
                   t_grid::T_RealVector)::Union{T_Real,
                                                T_RealArray}
    if t>=t_grid[end]
        k = length(t_grid)
    else
        t = max(t_grid[1], min(t_grid[end], t)) # Saturate to time grid
        k = get_interval(t, t_grid)
    end
    f_t = @k(f_cps)
    return f_t
end

""" Straight-line interpolation between two points.

Compute a straight-line interpolation between an initial and a final vector, on
a grid of N points.

# Arguments
    v0: starting vector.
    vf: ending vector.
    N: number of vectors in between (including endpoints).

# Returns
    v: the resulting interpolation (a matrix, k-th column is the k-th vector,
       k=1,...,N).
"""
function straightline_interpolate(
    v0::T_RealVector,
    vf::T_RealVector,
    N::T_Int)::T_RealMatrix

    # Initialize
    nv = length(v0)
    v = zeros(nv,N)

    # Interpolation grid
    times = LinRange(0.0, 1.0, N)
    t_endpts = [@first(times), @last(times)]
    v_endpts = hcat(v0, vf)

    for k = 1:N
        @k(v) = linterp(@k(times), v_endpts, t_endpts)
    end

    return v
end

""" Spherical linear interpolation between two quaternions.

See Section 2.7 of [Sola2017].

References:

```
@article{Sola2017,
  author    = {Joan Sola},
  title     = {Quaternion kinematics for the error-state Kalman filter},
  journal   = {CoRR},
  year      = {2017},
  url       = {http://arxiv.org/abs/1711.02508}
}
```

# Arguments
    q0: the starting quaternion.
    q1: the final quaternion.
    τ: interpolation mixing factor, in [0, 1]. When τ=0, q0 is returned; when
        τ=1, q1 is returned.

# Returns
    qt: the interpolated quaternion between q0 and q1.
"""
function slerp_interpolate(q0::T_Quaternion,
                           q1::T_Quaternion,
                           τ::T_Real)::T_Quaternion
    τ = max(0.0, min(1.0, τ))
    Δq = q1*q0' # Error quaternion correcting q0 to q1
    Δα, Δa = Log(Δq)
    Δq_t = T_Quaternion(τ*Δα, Δa)
    qt = Δq_t*q0
    return qt
end

""" Get the value of a continuous-time trajectory at time t.

# Arguments
    traj: the trajectory.
    t: the evaluation time.

# Returns
    x: the trajectory value at time t.
"""
function sample(traj::T_ContinuousTimeTrajectory,
                t::T_Real)::Union{T_Real,
                                  T_RealArray}

    if traj.interp==:linear
        x = linterp(t, traj.x, traj.t)
    elseif traj.interp==:zoh
        x = zohinterp(t, traj.x, traj.t)
    end

    return x
end

""" Classic Runge-Kutta integration over a provided time grid.

# Arguments
    f: the dynamics function.
    x0: the initial condition.
    tspan: the discrete time grid over which to integrate.
    full: (optional) whether to return the full trajectory, or just
        the final point.
    actions: (optional) actions to perform on the state after the
        numerical integration state update.

# Returns
    X: the integrated trajectory (final point, or full).
"""
function rk4(f::Function,
             x0::T_RealVector,
             tspan::T_RealVector;
             full::T_Bool=false,
             actions::T_SIA=T_SIA(undef, 0))::Union{T_RealVector,
                                                    T_RealMatrix}
    X = _helper__rk4_generic(f, x0; tspan=tspan, full=full, actions=actions)
    return X
end

""" Classic Runge-Kutta integration given a final time and time step.

# Arguments
    f: the dynamics function.
    x0: the initial condition.
    tf: the final time.
    h: the discrete time step.
    full: (optional) whether to return the full trajectory, or just
        the final point.
    actions: (optional) actions to perform on the state after the
        numerical integration state update.

# Returns
    X: the integrated trajectory (final point, or full).
"""
function rk4(f::Function,
             x0::T_RealVector,
             tf::T_Real,
             h::T_Real;
             full::T_Bool=false,
             actions::T_SIA=T_SIA(undef, 0))::Union{T_RealVector,
                                                    T_RealMatrix}
    # Define the scaled dynamics and time step
    F = (τ::T_Real, x::T_RealVector) -> tf*f(tf*τ, x)
    h /= tf

    # Integrate
    X = _helper__rk4_generic(F, x0; h=h, full=full, actions=actions)

    return X
end

""" Integrate a discrete signal using trapezoid rule.

# Arguments
    f: the function values on a discrete grid.
    grid: the discrete grid.

# Returns
    F: the numerical integral of the f signal.
"""
function trapz(f::AbstractVector, grid::T_RealVector)
    N = length(grid)
    F = 0.0
    for k = 1:N-1
        δ = @kp1(grid)-@k(grid)
        F += 0.5*δ*(@kp1(f)+@k(f))
    end
    return F
end

""" Project an ellipsoid onto a subset of its axes.

# Arguments
    E: the ellipsoid to be projected.
    ax: array of the axes onto which to project.

# Returns
    E_prj: the projected ellipsoid.
"""
function project(E::T_Ellipsoid, ax::T_IntVector)::T_Ellipsoid
    # Parameters
    n = length(E.c)
    m = length(ax)

    # Projection matrix onto lower-dimensional space
    P = zeros(m, n)
    for i=1:m
        P[i, ax[i]] = 1.0
    end

    # Do the projection
    Hb = P*E.H
    F = svd(Hb)
    H_prj = F.U*diagm(F.S)
    c_prj = P*E.c
    E_prj = T_Ellipsoid(H_prj, c_prj)

    return E_prj
end

""" Quaternion indexing.

# Arguments
    q: the quaternion.
    i1: the index.

# Returns
    v: the value.
"""
function getindex(q::T_Quaternion, i1::T_Int)::T_Real
    if i1<0 || i1>4
        err = ArgumentError("ERROR: quaternion index out of bounds.")
        throw(err)
    end

    if i1<=3
        v = q.v[i1]
    else
        v = q.w
    end

    return v
end

""" Skew-symmetric matrix from a 3-element vector.

# Arguments
    v: the vector.

# Returns
    S: the skew-symmetric matrix.
"""
function skew(v::T_RealVector)::T_RealMatrix
    S = zeros(3, 3)
    S[1,2], S[1,3], S[2,3] = -v[3], v[2], -v[1]
    S[2,1], S[3,1], S[3,2] = -S[1,2], -S[1,3], -S[2,3]
    return S
end

""" Skew-symmetric matrix from a quaternion.

# Arguments
    q: the quaternion.
    side: (optional) either :L or :R. In which case:
      :L : q*p and let [q]*p, return the 4x4 matrix [q]
      :R : q*p and let [p]*q, return the 4x4 matrix [p]

# Returns
    S: the skew-symmetric matrix.
"""
function skew(q::T_Quaternion, side::T_Symbol=:L)::T_RealMatrix
    S = T_RealMatrix(undef, 4, 4)
    S[1:3, 1:3] = q.w*I(3)+((side==:L) ? 1 : -1)*skew(q.v)
    S[1:3, 4] = q.v
    S[4, 1:3] = -q.v
    S[4, 4] = q.w
    return S
end

""" Quaternion multiplication.

# Arguments
    q: the first quaternion.
    p: the second quaternion.

# Returns
    r: the resultant quaternion, r=q*p.
"""
function *(q::T_Quaternion, p::T_Quaternion)::T_Quaternion
    r = T_Quaternion(skew(q)*vec(p))
    return r
end

""" Quaternion multiplication by a pure quaternion (a vector).

# Arguments
    q: the first quaternion.
    p: the second quaternion, as a 3-element vector.

# Returns
    r: the resultant quaternion, r=q*p.
"""
function *(q::T_Quaternion, p::T_RealVector)::T_Quaternion
    if length(p)!=3
        err = ArgumentError("ERROR: p must be a vector in R^3.")
        throw(err)
    end
    r = q*T_Quaternion(p)
    return r
end

""" Same as above function, but reverse order
"""
function *(q::T_RealVector, p::T_Quaternion)::T_RealVector
    if length(q)!=3
        err = ArgumentError("ERROR: q must be a vector in R^3.")
        throw(err)
    end
    r = T_Quaternion(q)*p
    return r
end

""" Quaternion conjugate.

# Arguments
    q: the original quaternion.

# Returns
    p: the conjugate of the quaternion.
"""
function adjoint(q::T_Quaternion)::T_Quaternion
    p = T_Quaternion(-q.v, q.w)
    return p
end

""" Logarithmic map of a unit quaternion.

It returns the axis and angle associated with the quaternion operator.
We assume that a unit quaternion is passed in (no checks are run to verify
this).

# Arguments
    q: the unit quaternion.

# Returns
    α: the rotation angle (in radiancs).
    a: the rotation axis.
"""
function Log(q::T_Quaternion)::Tuple{T_Real, T_RealVector}
    nrm_qv = norm(q.v)
    α = 2*atan(nrm_qv, q.w)
    a = q.v/nrm_qv
    return α, a
end

""" Compute the direction cosine matrix associated with a quaternion.

# Arguments
    q: the quaternion.

# Returns
    R: the 3x3 direction cosine matrix.
"""
function dcm(q::T_Quaternion)::T_RealMatrix
    R = (skew(q', :R)*skew(q))[1:3, 1:3]
    return R
end

""" Compute Euler angle sequence associated with a quaternion.

Use the Z-Y'-X'' convention (Tait-Bryan angles [1]):
  0. Begin with the world coordinate system {X,Y,Z};
  1. First, rotate ("yaw") about Z. Obtain {X',Y',Z'};
  2. Next, rotate ("pitch") about Y'. Obtain {X'',Y'',Z''};
  2. Finally, rotate ("roll") about X''. Obtain the body coordinate system.

This returns an **active** rotation matrix R. Specifically, given a vector x in
the world coordinate system, R*x=x' which is the vector rotated by R, still
expressed in the world coordinate system. In particular, R*(e_i) gives the
principal axes of the rotated coordinate system, expressed in the world
coordinate system. If you want a passive rotation, use transpose(R).

References:

[1] https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles

# Arguments
    q: the quaternion.

# Returns
    v: a 3-tuple of the angles (yaw, pitch, roll) (in radians).
"""
function rpy(q::T_Quaternion)::Tuple{T_Real, T_Real, T_Real}
    R = dcm(q)
    pitch = acos(max(0.0, min(1.0, sqrt(R[1, 1]^2+R[2, 1]^2))))
    roll = atan(R[3, 2], R[3, 3])
    yaw = atan(R[2, 1], R[1, 1])
    return yaw, pitch, roll
end

""" Convert quaternion to vector form.

# Arguments
    q: the quaternion.

# Returns
    q_vec: the quaternion in vector form, scalar last.
"""
function vec(q::T_Quaternion)::T_RealVector
    q_vec = [q.v; q.w]
    return q_vec
end

""" Evaluate ellipsoid level set value at location.

# Arguments
    r: location at which to evaluate ellipsoid value.

# Returns
    y: the level set value.
"""
function (E::T_Ellipsoid)(r::T_RealVector)::T_Real
    y = norm(E.H*(r-E.c))
    return y
end

""" Ellipsoid gradient at location.

# Arguments
    r: location at which to evaluate ellipsoid gradient.

# Returns
    g: the gradient value.
"""
function ∇(E::T_Ellipsoid, r::T_RealVector)::T_RealVector
    g = (E.H'*E.H)*(r-E.c)/E(r)
    return g
end

""" Check if a point is contained in the hyperrectangle.

# Arguments
    H: the hyperrectangle set.
    r: the point.

# Returns
    Boolean true iff r∈H.
"""
function contains(H::T_Hyperrectangle, r::T_RealVector)::T_Bool
    return all(H.l .<= r .<= H.u)
end

"""
    logsumexp(f[, ∇f][; t])

Compute log-sum-exp function value and its gradient. Numerically stable
implementation based on [1]. The log-sum-exp function is:

` L(x) = 1/t*log( ∑_i^{m} exp(t*f_i(x)) ). `

As t increases, this becomes an increasingly accurate approximation of `max_i
f_i(x)`.

References:

[1] https://leimao.github.io/blog/LogSumExp/

# Arguments
- `f`: the function in the exponent.
- `∇f`: (optional) the gradient of the function.

# Keywords
- `t`: (optional) the homotopy parameter.

# Returns
- `L`: the log-sum-exp function value.
- `∇L`: the log-sum-exp function gradient. Only returned in ∇f provided.
"""
function logsumexp(
    f::T_RealVector,
    ∇f::Union{Nothing, Vector{T_RealVector}}=nothing;
    t::T_Real=1.0)::Union{Tuple{T_Real, T_RealVector},
                          T_Real}

    n = length(f)

    # Value
    a = maximum(t*f)
    sumexp = sum([exp(t*f[i]-a) for i=1:n])
    logsumexp = a+log(sumexp)
    L = logsumexp/t

    # Gradient
    if !isnothing(∇f)
        ∇L = sum([∇f[i]*exp(t*f[i]-a) for i=1:n])/sumexp
        return L, ∇L
    end

    return L
end

"""
    sigmoid(f[, ∇f][; κ])

Compute a sigmoid function. The sigmoid represence a logical "OR" combination
of its argument, in the sense that its value tends to one as any argument
becomes more positive.

# Arguments
- `f`: the function in the exponent.
- `∇f`: (optional) the gradient of the function.

# Keywords
- `κ`: (optional) the sharpness parameter.

# Returns
- `σ`: the sigmoid function value.
- `∇σ`: the sigmoid function gradient. Only returned in ∇f provided.
"""
function sigmoid(f::T_RealVector,
                 ∇f::Union{Nothing, Vector{T_RealVector}}=nothing;
                 κ::T_Real=1.0)::Union{Tuple{T_Real, T_RealVector},
                                       T_Real}
    L = logsumexp(f, ∇f; t=κ)
    if !isnothing(∇f)
        L, ∇L = L
    end
    σ = 1-1/(1+exp(κ*L))
    if !isnothing(∇f)
        ∇σ = κ*exp(κ*L)*(1-σ)^2*∇L
        return σ, ∇σ
    end
    return σ
end

"""
    indicator(f[, ∇f][; κ1, κ2])

Description.

# Arguments
- `f`: the function in the exponent.
- `∇f`: (optional) the gradient of the function.

# Keywords
- `κ1`: (optional) sigmoid sharpness parameter.
- `κ2`: (optional) normalization sharpness parameter.

# Returns
- `δ`: the sigmoid function value.
- `∇δ`: the sigmoid function gradient. Only returned in ∇f provided.
"""
function indicator(f::T_RealVector,
                   ∇f::Union{Nothing, Vector{T_RealVector}}=nothing;
                   κ1::T_Real=1.0,
                   κ2::T_Real=1.0)::Union{Tuple{T_Real, T_RealVector},
                                          T_Real}
    σ = sigmoid(f, ∇f; κ=κ1)
    if !isnothing(∇f)
        σ, ∇σ = σ
    end
    γ = exp(-κ2*κ1)
    δ = γ+(1-γ)*σ
    if !isnothing(∇f)
        ∇δ = (1-γ)*∇σ
        return δ, ∇δ
    end
    return δ
end

"""
    predicates(p1[, p2, ..., pN][; κ1, κ2])

A smooth logical OR.

# Arguments
- `predicates`: (optional) the predicates to be composed as logical
  "OR". Provide either a list of reals `f` or a list of (real, vector) tuples
  `(f, ∇f)`.

# Keywords
- `κ1`: (optional) sigmoid sharpness parameter.
- `κ2`: (optional) normalization sharpness parameter.

# Returns
- `OR`: the smooth OR value (1≡true).
- `∇OR`: the smooth OR gradient. Only returned in `∇f` provided.
"""
function or(predicates...;
            κ1::T_Real=1.0,
            κ2::T_Real=1.0)::Union{Tuple{T_Real, T_RealVector},
                                   T_Real}
    nargs = length(predicates)
    if typeof(predicates[1])<:Tuple
        f = T_RealVector([p[1] for p in predicates])
        ∇f = [p[2] for p in predicates]
        OR, ∇OR = indicator(f, ∇f; κ1=κ1, κ2=κ2)
        return OR, ∇OR
    else
        f = T_RealVector([p for p in predicates])
        OR = indicator(f; κ1=κ1, κ2=κ2)
        return OR
    end
end

"""
    print(row, table)

Print row of table.

# Arguments
- `row`: table row specification.
- `table`: the table specification.
"""
function print(row::Dict{T_Symbol, T}, table::T_Table)::Nothing where {T}
    # Assign values to table columns
    values = fill("", length(table.headings))
    for (k, v) in row
        val_fmt = table.fmt[k]
        values[table.sorting[k]] = @eval @sprintf($val_fmt, $v)
    end

    if table.__head_print==true
        table.__head_print = false
        # Print the columnd headers
        top = @eval @printf($(table.row), $(table.headings)...)
        println()

        _helper__table_print_hrule(table)
    end

    msg = @eval @printf($(table.row), $values...)
    println()

    return nothing
end

""" Compute the relative cost improvement (as a string to be put into a table).

# Arguments
    J_new: next cost.
    J_old: old cost.

# Returns
    ΔJ: the relative cost improvement.
"""
function cost_improvement_percent(J_new::T_Real, J_old::T_Real)::T_String
    if isnan(J_old)
        ΔJ = ""
    else
        ΔJ = (J_old-J_new)/abs(J_old)*100
        _ΔJ = @sprintf("%.2f", ΔJ)
        if length(_ΔJ)>8
            fmt = string("%.", (ΔJ>0) ? 2 : 1, "e")
            ΔJ = @eval @sprintf($fmt, $ΔJ)
        else
            ΔJ = _ΔJ
        end
    end

    return ΔJ
end

""" Reset table printing.

This will make the columnd headings be printed again.

# Arguments
    table: the table specification.
"""
function reset(table::T_Table)::Nothing
    table.__head_print = true
    return nothing
end

""" Plot a constant y-value bound keep-out zone of a timeseries.

Supposedly to show a minimum or a maximum of a quantity on a time history plot.

# Arguments
    ax: the figure axis object.
    x_min: the left-most value.
    x_max: the right-most value.
    y_bnd: the bound value.
    height: the "thickness" of the keep-out slab on the plot.
"""
function plot_timeseries_bound!(ax::PyPlot.PyObject,
                                x_min::T_Real,
                                x_max::T_Real,
                                y_bnd::T_Real,
                                height::T_Real)::Nothing

    y_other = y_bnd+height
    x = [x_min, x_max, x_max, x_min, x_min]
    y = [y_bnd, y_bnd, y_other, y_other, y_bnd]

    fc = parse(RGB, "#db6245")
    ax.fill(x, y,
            facecolor=rgb2pyplot(fc, a=0.5),
            edgecolor="none")

    ax.plot([x_min, x_max],
            [y_bnd, y_bnd],
            color="#db6245",
            linewidth=1.75,
            linestyle="--",
            dashes=(2, 3),
            solid_capstyle="round",
            dash_capstyle="round")

    return nothing
end

""" Plot a variable y-value bound keep-out zone of a timeseries.

Supposedly to show a minimum or a maximum of a quantity on a time history plot.

# Arguments
    ax: the figure axis object.
    x_bnd: the x-axis values.
    y_bnd: the bound values.
    height: the "thickness" of the keep-out slab on the plot. Above max(y) if
        positive or below min(y) if negative.
"""
function plot_timeseries_bound!(ax::PyPlot.PyObject,
                                x_bnd::T_RealVector,
                                y_bnd::T_RealVector,
                                height::T_Real)::Nothing

    bnd_top = (height>=0) ? maximum(y_bnd)+height : minimum(y_bnd)+height
    X = vcat(x_bnd, reverse(x_bnd))
    Y = vcat(y_bnd, fill(bnd_top, length(y_bnd)))

    fc = parse(RGB, "#db6245")
    ax.fill(X, Y,
            facecolor=rgb2pyplot(fc, a=0.5),
            edgecolor="none")

    ax.plot(x_bnd, y_bnd,
            color="#db6245",
            linewidth=1.75,
            linestyle="--",
            dashes=(2, 3),
            solid_capstyle="round",
            dash_capstyle="round")

    return nothing
end

"""
    plot_ellipsoids!(ax, E[, axes][; label])

Draw ellipsoids on the currently active plot.

# Arguments
- `ax`: the figure axis object.
- `E`: array of ellipsoids.
- `axes`: (optional) which 2 axes to project onto.

# Keywords
- `label`: (optional) legend label.
"""
function plot_ellipsoids!(ax::PyPlot.PyObject,
                          E::Vector{T_Ellipsoid},
                          axes::T_IntVector=[1, 2];
                          label::Union{T_String,Nothing}=nothing)::Nothing
    θ = LinRange(0.0, 2*pi, 100)
    circle = hcat(cos.(θ), sin.(θ))'
    for i = 1:length(E)
        Ep = project(E[i], axes)
        vertices = Ep.H\circle.+Ep.c
        x, y = vertices[1, :], vertices[2, :]
        fc = parse(RGB, "#db6245")
        ax.fill(x, y,
                facecolor=rgb2pyplot(fc, a=0.5),
                edgecolor="#26415d",
                linewidth=1,
                label=(i==1) ? label : nothing)
    end
    return nothing
end

"""
    plot_prisms!(ax, H[, axes][; label])

Draw rectangular prisms on the current active plot.

# Arguments
- `ax`: the figure axis object.
- `H`: array of 3D hyperrectangle sets.
- `axes`: (optional) which 2 axes to project onto.

# Keywords
- `label`: (optional) legend label.
"""
function plot_prisms!(ax::PyPlot.PyObject,
                      H::Vector{T_Hyperrectangle},
                      axes::T_IntVector=[1, 2];
                      label::Union{T_String,Nothing}=nothing)::Nothing
    patch = nothing
    for i = 1:length(H)
        Hi = H[i]
        x, y = axes
        vertices = T_RealMatrix([Hi.l[x] Hi.u[x] Hi.u[x] Hi.l[x] Hi.l[x];
                                 Hi.l[y] Hi.l[y] Hi.u[y] Hi.u[y] Hi.l[y]])
        x, y = vertices[1, :], vertices[2, :]
        fc = parse(RGB, "#5da9a1")
        ax.fill(x, y,
                linewidth=1,
                facecolor=rgb2pyplot(fc, a=0.5),
                edgecolor="#427d77",
                label=(i==1) ? label : nothing)
    end
    return nothing
end

""" Optimization algorithm convergence and performance plot.

# Arguments
* `history`: SCP iteration data history.
* `name`: the example name.
"""
function plot_convergence(history, name::T_String)::Nothing

    # Common values
    algo = history.subproblems[1].algo
    clr = get_colormap()(1.0)

    # Compute concatenated solution vectors at each iteration
    num_iter = length(history.subproblems)
    xd = [vec(history.subproblems[i].sol.xd) for i=1:num_iter]
    ud = [vec(history.subproblems[i].sol.ud) for i=1:num_iter]
    p = [history.subproblems[i].sol.p for i=1:num_iter]
    Nnx = length(xd[1])
    Nnu = length(ud[1])
    np = length(p[1])
    X = T_RealMatrix(undef, Nnx+Nnu+np, num_iter)
    for i = 1:num_iter
        X[:, i] = vcat(xd[i], ud[i], p[i])
    end
    DX = T_RealVector([norm(X[:, i]-X[:, end])/norm(X[:, end])
                       for i=1:(num_iter-1)])
    no_change = findfirst((dx) -> dx==0, DX)
    if !isnothing(no_change)
        num_iter = no_change-1
        DX = DX[1:num_iter]
    else
        num_iter -= 1
    end
    iters = T_IntVector(1:num_iter)

    fig = create_figure((5, 6))

    if num_iter<=15
        xticks = iters
    else
        step = T_Int(ceil(num_iter/15))
        xticks = T_IntVector(1:step:num_iter)
    end

    # ..:: Convergence plot ::..

    ax = fig.add_subplot(211)

    ax.set_yscale("log")
    ax.grid(linewidth=0.3, alpha=0.5, axis="y", which="major")
    ax.grid(linewidth=0.2, alpha=0.5, axis="y", which="minor", linestyle="--")
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true, axis="x")
    ax.margins(x=0.04, y=0.04)
    ax.set_xticks(xticks)

    ax.set_xlabel("Iteration number")
    ax.set_ylabel(string("Distance from solution, ",
                         "\$\\frac{\\|X^i-X^*\\|_2}{",
                         "\\|X^*\\|_2}\$"))

    ax.plot(iters, DX,
            color=clr,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgewidth=0,
            clip_on=false,
            zorder=100)

    # Set y ticks
    # Based on: https://stackoverflow.com/a/64840431
    y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = matplotlib.ticker.LogLocator(
        base = 10.0, subs = (1:10)*0.1, numticks = 10)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax_top = ax

    # ..:: Timing performance plot ::..

    ax = fig.add_subplot(212)

    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true, axis="x")
    ax.margins(x=0.04, y=0.04)
    ax.set_xticks(xticks)

    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Time per iteration [s]")

    Yellow = "#f1d46a"
    Red = "#db6245"
    Blue = "#356397"
    DarkBlue = "#26415d"
    Green = "#5da9a1"

    labels = iters
    width = 0.3
    lw = 3.0+(num_iter-5)/35*(1.5-3.0) # Scaling that works visually well
    js = "bevel"
    darken_factor = 0.3
    darken = (c) -> rgb2pyplot(weighted_color_mean(
        1-darken_factor, parse(RGB, c), colorant"black"))

    spbms = history.subproblems[1:end-1]
    discretize = [spbm.timing[:discretize] for spbm in spbms]
    solve = [spbm.timing[:solve] for spbm in spbms]
    formulate = [spbm.timing[:formulate] for spbm in spbms]
    overhead = [spbm.timing[:overhead] for spbm in spbms]
    total = [spbm.timing[:total] for spbm in spbms]
    i_start = (length(solve)>1) ? 2 : 1
    ymax = maximum((formulate+discretize+solve+overhead)[i_start:end])*1.1

    for i = 1:2
        linew = (i==1) ? 0 : lw
        z = (i==1) ? 0 : -1
        lbl = (str) -> (i==1) ? str : nothing

        ax.bar(labels, discretize, width,
               label=lbl("Discretize"),
               color=Yellow,
               linewidth=linew,
               edgecolor=darken(Yellow),
               joinstyle=js,
               zorder=z)
        ax.bar(labels, solve, width,
               bottom=discretize,
               label=lbl("Solve"),
               color=Red,
               linewidth=linew,
               edgecolor=darken(Red),
               joinstyle=js,
               zorder=z)
        ax.bar(labels, formulate, width,
               bottom=discretize+solve,
               label=lbl("Formulate"),
               color=Green,
               linewidth=linew,
               edgecolor=darken(Green),
               joinstyle=js,
               zorder=z)
        ax.bar(labels, overhead, width,
               bottom=discretize+solve+formulate,
               label=lbl("Overhead"),
               color=DarkBlue,
               linewidth=linew,
               edgecolor=darken(DarkBlue),
               joinstyle=js,
               zorder=z)
    end

    ax.set_ylim(top=ymax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(reverse(handles), reverse(labels),
              framealpha=0.8, fontsize=8, loc="upper left")

    ax2 = ax.twinx()

    ax2_clr = Blue
    outline_w = 1.5

    ax2.set_ylabel("Cumulative time [s]", color=ax2_clr)
    ax2.tick_params(axis="y", colors=ax2_clr)
    ax2.spines["right"].set_edgecolor(ax2_clr)

    ax2.plot(iters, cumsum(total),
             color=ax2_clr,
             linewidth=2,
             marker="o",
             markersize=6,
             markeredgewidth=0,
             markeredgecolor="white",
             clip_on=false,
             zorder=100)
    ax2.plot(iters, cumsum(total),
             color="white",
             linewidth=2+outline_w,
             marker="o",
             markersize=6,
             markeredgewidth=outline_w,
             markeredgecolor="white",
             clip_on=false,
             zorder=99)

    ax_bot = ax

    fig.align_ylabels([ax_top, ax_bot])

    save_figure(@sprintf("%s_convergence", name), algo)

    return nothing
end

""" Get a plotting colormap.

The colormap is normalized to the [0, 1] interval.

# Returns
    cmap: a colormap object that can be queried for RGB color.
"""
function get_colormap()::PyPlot.PyObject
    cmap = plt.get_cmap("inferno_r")
    nrm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    clr_query = matplotlib.cm.ScalarMappable(norm=nrm, cmap=cmap)
    cmap = (f::T_Real) -> clr_query.to_rgba(f)[1:3]
    return cmap
end

""" Convert RGB color object to a tuple that PyPlot accepts.

# Arguments
    c: the RGB color object.
    a: (optional) the alpha (opacity) channel value.

# Returns
    t: a 4-tuple (R, G, B, A) for PyPlot.
"""
function rgb2pyplot(c::T; a::Real=1)::Tuple{Real, Real,
                                            Real, Real} where {T<:RGB}
    r, g, b = T_Real(c.r), T_Real(c.g), T_Real(c.b)
    t = (r, g, b, a)
    return t
end

"""
    create_figure(size)

Create an empty figure for plotting.

# Arguments
- `size`: the figure size (width, height).

# Returns
- `fig`: the figure object.
"""
function create_figure(size::Tuple{T, V})::Figure where {T<:Real, V<:Real}

    # Set plot parameters
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    rcParams["text.usetex"] = true
    rcParams["font.family"] = "sans-serif"
    rcParams["axes.labelsize"] = 14
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12
    rcParams["text.latex.preamble"] = string("\\usepackage{sansmath}",
                                             "\\sansmath")

    plt.ioff()

    fig = plt.figure(figsize=size)

    plt.clf()

    return fig
end

"""
    save_figure(filename, algo[; tmp, path])

Save the current figure to a PDF file. The filename is prepended with the name
of the SCP algorithm used for the solution.

# Arguments
- `filename`: the filename of the figure.
- `algo`: the SCP algorithm string (format "<SCP_ALGO> (backend: <CVX_ALGO>)").

# Keywords
- `tmp`: (optional) whether this is a temporary file.
- `path`: (optional) where to save the figure.
"""
_helper__tight_layout_applied = false
function save_figure(filename::T_String, algo::T_String;
                     tmp::T_Bool=false,
                     path::Union{T_String, Nothing}=nothing)::Nothing

    global _helper__tight_layout_applied

    algo = lowercase(split(algo, " "; limit=2)[1])
    path = isnothing(path) ? "../../figures/" : path

    # Apply tight layout, only do this **once** per figure
    if !_helper__tight_layout_applied
        plt.tight_layout()
        _helper__tight_layout_applied = true
    end

    # Save figure
    if !tmp
        plt.savefig(@sprintf("%s%s_%s.pdf", path, algo, filename),
                    bbox_inches="tight", pad_inches=0.01, facecolor=zeros(4))
        plt.close()
        _helper__tight_layout_applied = false # reset
    else
        plt.savefig(@sprintf("/tmp/%s_%s.pdf", algo, filename),
                    bbox_inches="tight", pad_inches=0.01, facecolor=zeros(4))
    end

    return nothing
end

"""
    set_axis_equal(ax, lims)

Set axis limits with an equal aspect ratio (i.e. circles appear as circles).

# Arguments
- `ax`: the axis object.
- `lims`: a four-tuple of scalars (xmin,xmax,ymin,ymax). At least one of these
  has to be `missing`, which means the bound of that limit is decided by the
  scaling amount.
"""
function set_axis_equal(
    ax::PyPlot.PyObject,
    lims::Tuple{Union{Real, Missing},
                Union{Real, Missing},
                Union{Real, Missing},
                Union{Real, Missing}})::Nothing

    ax.axis("equal")
    save_figure("scp_tmp_fig", "", tmp=true)
    x_rng = ax.get_xlim()
    y_rng = ax.get_ylim()
    ar = (y_rng[2]-y_rng[1])/(x_rng[2]-x_rng[1])

    ax.axis("auto")
    xmin, xmax, ymin, ymax = lims
    if ismissing(xmin)
        xmin = xmax-(ymax-ymin)/ar
    elseif ismissing(xmax)
        xmax = xmin+(ymax-ymin)/ar
    elseif ismissing(ymin)
        ymin = ymax-ar*(xmax-xmin)
    elseif ismissing(ymax)
        ymax = ymin+ar*(xmax-xmin)
    end
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return nothing
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Update the state using a single Runge-Kutta integration update.

# Arguments
    f: the dynamics function.
    x: the current state.
    t: the current time.
    tp: the next time.

# Returns
    xp: the next state.
"""
function _helper__rk4_core_step(f::Function,
                                x::T_RealVector,
                                t::T_Real,
                                tp::T_Real)::T_RealVector

    # Time step length
    h = tp-t

    # The Runge-Kutta update
    k1 = f(t,x)
    k2 = f(t+h/2,x+h/2*k1)
    k3 = f(t+h/2,x+h/2*k2)
    k4 = f(t+h,x+h*k3)
    xp = x+h/6*(k1+2*k2+2*k3+k4)

    return xp
end

""" Classic Runge-Kutta integration.

Interate a system of ordinary differential equations over a time interval. The
`tspan` and `h` arguments are mutually exclusive: one and exactly one of them
must be provided.

# Arguments
    f: the dynamics function, with call signature f(t, x) where t is the
        current time and x is the current state.
    x0: the initial condition.
    tspan: (optional) the time grid over which to integrate.
    h: (optional) the time step of a uniform time grid over which to integrate.
        If provided, a [0, 1] integration interval is assumed. If the time step
        does not split the [0, 1] interval into an integral number of temporal
        nodes, then the largest time step <=h is used.
    full: (optional) whether to return the complete trajectory (if false,
        return just the endpoint).
    actions: (optional) actions to perform on the state after the numerical
        integration state update.

# Returns
    X: the integration result (final point, or full trajectory, depending on
        the argument `full`).
"""
function _helper__rk4_generic(f::Function,
                              x0::T_RealVector;
                              tspan::Union{T_RealVector, Nothing}=nothing,
                              h::Union{T_Real, Nothing}=nothing,
                              full::T_Bool=false,
                              actions::T_SIA=T_SIA(undef, 0))::Union{
                                  T_RealVector,
                                  T_RealMatrix}

    # Check that one and only one of the arguments tspan and h is passed
    if !xor(isnothing(tspan), isnothing(h))
        err = ArgumentError(string("ERROR: rk4 accepts one and only one",
                                   " of tspan or h."))
        throw(err)
    end

    if isnothing(tspan)
        # Construct a uniform tspan such that the time step is the largest
        # possible value <=h
        N = ceil(T_Int, 1.0+1.0/h)
        tspan = LinRange(0.0, 1.0, N)
    end

    # Initialize the data
    n = length(x0)
    N = length(tspan)
    if full
        X = T_RealMatrix(undef, n, N)
        @first(X) = copy(x0)
    else
        X = copy(x0)
    end

    # Integrate the function
    for k = 2:N
        if full
            @k(X) = _helper__rk4_core_step(
                f, @km1(X), @km1(tspan), @k(tspan))
            # Update actions
            for act in actions
                @k(X)[act[1]] = act[2](@k(X)[act[1]])
            end
        else
            X = _helper__rk4_core_step(
                f, X, @km1(tspan), @k(tspan))
            # Update actions
            for act in actions
                X[act[1]] = act[2](X[act[1]])
            end
        end
    end

    return X
end

""" Print table row horizontal separator line.

# Arguments
    table: the table specification.
"""
function _helper__table_print_hrule(table::T_Table)::Nothing
    hrule = ""
    num_cols = length(table.__colw)
    for i = 1:num_cols
        hrule = string(hrule, repeat("-", table.__colw[i]))
        if i < num_cols
            hrule = string(hrule, "-+-")
        end
    end
    println(hrule)

    return nothing
end
