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

import Base: vec, adjoint, *, print

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

"""
    add_conic_constraint!(pbm, cone)

Add a conic constraint to the optimization problem.

# Arguments
- `pbm`: the optimization problem structure.
- `cone`: the conic constraint structure.

# Returns
- `constraint`: the conic constraint reference.
"""
function add_conic_constraint!(
    pbm::Model,
    cone::T)::T_Constraint where {T<:T_ConvexConeConstraint}

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

"""
    fixed_cone(cone)

Check if the cone is fixed (i.e. just numerical values, not used inside an
optimization).

# Arguments
- `cone`: the cone object.

# Returns
- `is_fixed`: true if the cone is purely numerical.
"""
function fixed_cone(cone::T_ConvexConeConstraint)::T_Bool
    z_real = typeof(cone.z)<:Array{T} where {T<:Real}
    is_fixed = typeof(cone.z)==T_RealVector
    return is_fixed
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

"""
    h(x)

Get the homotopy parameter value.

# Arguments
- `x`: a number between [0,1] which maps to the homotopy value. Should increase
  linearly, and the homotopy will take care of mapping to a nonlinear function.

# Returns
- `h`: homotopy value for x.
"""
function (hom::T_Homotopy)(x::T_Real)::T_Real
    h = log(1/hom.ε-1)/(hom.ρ^(x)*hom.δ_max)
    return h
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
        coeff = exp(κ*L+2*log(1-σ))
        ∇σ = κ*coeff*∇L
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
    or(p1[, p2, ..., pN][; κ1, κ2])

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
