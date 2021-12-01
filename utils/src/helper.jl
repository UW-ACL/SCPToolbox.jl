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

using Printf

export skew, get_interval, linterp, zohinterp, diracinterp,
    straightline_interpolate, rk4, trapz, ∇trapz, logsumexp, or, squeeze,
    convert_units, homtransf, hominv, homdisp, homrot, make_indent,
    golden, c2d, test_heading

export @preprintf

# ..:: Globals ::..

const T = Types
const RealValue = T.RealTypes
const RealArray = T.RealArray
const RealVector = T.RealVector
const RealMatrix = T.RealMatrix
const Optional = Types.Optional

# ..:: Methods ::..

""" Skew-symmetric matrix from a 3-element vector.

# Arguments
    v: the vector.

# Returns
    S: the skew-symmetric matrix.
"""
function skew(v::RealVector)::RealMatrix
    S = zeros(3, 3)
    S[1,2], S[1,3], S[2,3] = -v[3], v[2], -v[1]
    S[2,1], S[3,1], S[3,2] = -S[1,2], -S[1,3], -S[2,3]
    return S
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
function get_interval(x::RealValue, grid::RealVector)::Int
    k = sum(x.>grid)
    if k==0
	k = 1
    end
    return k
end

"""
    linterp(t, f_cps, t_grid)

Linear interpolation on a grid. Linearly interpolate a discrete function on a
time grid. In other words, get the function value assuming that it is a
continuous and piecewise affine function.

# Arguments
- `t`: the time at which to get the function value.
- `f_cps`: the control points of the function, stored as columns of a matrix.
- `t_grid`: the discrete time nodes.

# Returns
- `f_t`: the function value at time t.
"""
function linterp(t::RealValue,
                 f_cps::RealArray,
                 t_grid::RealVector)::Union{
                     RealValue, RealArray}
    t = max(t_grid[1], min(t_grid[end], t)) # Saturate to time grid
    k = get_interval(t, t_grid)
    c = (t_grid[k+1]-t)/(t_grid[k+1]-t_grid[k])
    dimfill = fill(:, ndims(f_cps)-1)
    f_t = c*f_cps[dimfill..., k]+(1-c)*f_cps[dimfill..., k+1]
    return f_t
end

"""
    zohinterp(t, f_cps, t_grid)

Zeroth-order hold interpolation on a grid. Previous neighbor interpolation. The
interpolated value at a query point is the value at the previous sample grid
point.

# Arguments
    t: the time at which to get the function value.
    f_cps: the control points of the function, stored as columns of a matrix.
    t_grid: the discrete time nodes.

# Returns
    f_t: the function value at time t.
"""
function zohinterp(t::RealValue,
                   f_cps::RealArray,
                   t_grid::RealVector)::Union{
                       RealValue, RealArray}
    if t>=t_grid[end]
        k = length(t_grid)
    else
        t = max(t_grid[1], min(t_grid[end], t)) # Saturate to time grid
        k = get_interval(t, t_grid)
    end
    dimfill = fill(:, ndims(f_cps)-1)
    f_t = f_cps[dimfill..., k]
    return f_t
end

"""
    diracinterp(t, f_cps, t_grid)

Impulse signal interpolation on a grid. If the time `t` does not land exactly
on a temporal grid node, return zero. Otherwise, return the signal value (which
would multiply the unit area of a Dirac spike, if it is an impulse signal).

# Arguments
- `t`: the time at which to get the function value.
- `f_cps`: the control points of the function, stored as columns of a matrix.
- `t_grid`: the discrete time nodes.

# Returns
- `f_t`: the function value at time t.
"""
function diracinterp(t::RealValue,
                     f_cps::RealArray,
                     t_grid::RealVector)::Union{
                         RealValue, RealArray}
    if t>=t_grid[end]
        k = length(t_grid)
    else
        t = max(t_grid[1], min(t_grid[end], t)) # Saturate to time grid
        k = findall(t_grid.==t)
        if isempty(k)
            f_t = zeros(size(f_cps)[1:end-1])
            return f_t
        else
            k = k[1]
        end
    end
    dimfill = fill(:, ndims(f_cps)-1)
    f_t = f_cps[dimfill..., k]
    return f_t
end

"""
    straightline_interpolate(v0, vf, N)

Straight-line interpolation between two points. Compute a straight-line
interpolation between an initial and a final vector, on a grid of N points.

# Arguments
- `v0`: starting vector.
- `vf`: ending vector.
- `N`: number of vectors in between (including endpoints).

# Returns
- `v`: the resulting interpolation (a matrix, k-th column is the k-th vector,
  k=1,...,N).
"""
function straightline_interpolate(
    v0::RealVector,
    vf::RealVector,
    N::Int)::RealMatrix

    # Initialize
    nv = length(v0)
    v = zeros(nv,N)

    # Interpolation grid
    times = LinRange(0.0, 1.0, N)
    t_endpts = [times[1], times[end]]
    v_endpts = hcat(v0, vf)

    for k = 1:N
        v[:, k] = linterp(times[k], v_endpts, t_endpts)
    end

    return v
end

"""
    c2d(A, B, p, Δt)

Discretize continuous-time linear time invartiant dynamics at a Δt time step
using zeroth-order hold (ZOH). This effectively convert the following system:

```math
\\dot x(t) = A x(t)+B u(t)+p,
```

to the following discrete-time system:

```math
x_{k+1} = A_d x_k+B_d u_k+p_d
```

# Arguments
- `A`: the state coefficient matrix.
- `B`: the input coefficient matrix.
- `p`: a constant exogenous disturbance vector.
- `Δt`: the time step.

# Returns
- `Ad`: the discrete-time state coefficient matrix.
- `Bd`: the discrete-time input coefficient matrix.
- `pd`: the discrete-time exogenous disturbance time step.
"""
function c2d(A::RealMatrix,
             B::RealMatrix,
             p::RealVector,
             Δt::RealValue)::Tuple{RealMatrix, RealMatrix, RealVector}

    n = size(A, 1)
    m = size(B, 2)

    M = exp(RealMatrix([A B p; zeros(m+1, n+m+1)])*Δt)

    Ad = M[1:n,1:n]
    Bd = M[1:n,n+1:n+m]
    pd = M[1:n,n+m+1]

    return Ad, Bd, pd
end

"""
    golden(f, a, b[; tol])

Golden search for minimizing a unimodal function ``f(x)`` on the interval `[a,
b]` to within a prescribed tolerance in ``x``. Implementation is based on [1].

References

[1] M. J. Kochenderfer and T. A. Wheeler, Algorithms for
Optimization. Cambridge, Massachusetts: The MIT Press, 2019.

# Arguments
- `f`: oracle with call signature `v=f(x)` where `v` is saught to be minimized.
- `a`: search domain lower bound.
- `b`: search domain upper bound.

# Keywords
- `tol`: (optional) tolerance in terms of maximum distance that the minimizer
  `x∈[a,b]` is away from `a` or `b`.
- `verbose`: (optional) print the golden searchprogress.

# Returns
- `sol`: a tuple where `s[1]` is the argmin and `s[2]` is the min.
"""
function golden(f::Function, a::RealValue, b::RealValue;
                tol::RealValue=1e-3,
                verbose::Bool=true)::Tuple{RealValue, RealValue}

    φ = (1+sqrt(5))/2
    n = ceil(log((b-a)/tol)/log(φ)+1)
    ρ = φ-1
    d = ρ*b+(1-ρ)*a
    yd = f(d)

    if verbose
        @printf("Golden search bracket:\n")
    end

    for i = 1:n-1

        c = ρ*a+(1-ρ)*b
        yc = f(c)

        if yc<yd
            b,d,yd = d,c,yc
        else
            a,b = b,c
        end

        bracket = sort([a,b,c,d])

        if verbose
            @printf("%-10.3e | %-10.3e | %-10.3e | %-10.3e\n", bracket...)
        end
    end

    x_sol = b
    sol = (x_sol, f(x_sol))

    return sol
end

"""
    rk4(f, x0, tspan[; full, actions])

Classic Runge-Kutta integration over a provided time grid.

# Arguments
- `f`: the dynamics function.
- `x0`: the initial condition.
- `tspan`: the discrete time grid over which to integrate.
- `full`: (optional) whether to return the full trajectory, or just the final
  point.
- `actions`: (optional) actions to perform on the state after the numerical
  integration state update.

# Returns
- `X`: the integrated trajectory (final point, or full).
"""
function rk4(f::T.Func,
             x0::RealVector,
             tspan::RealVector;
             full::Bool=false,
             actions::T.SpecialIntegrationActions=
                 T.SpecialIntegrationActions(undef, 0))::Union{
                     RealVector, RealMatrix}
    X = rk4_generic(f, x0; tspan=tspan, full=full, actions=actions)
    return X
end

"""
    rk4(f, x0, tf, h[; full, actions])

Classic Runge-Kutta integration given a final time and time step.

# Arguments
- `f`: the dynamics function.
- `x0`: the initial condition.
- `tf`: the final time.
- `h`: the discrete time step.
- `full`: (optional) whether to return the full trajectory, or just the final
  point.
- `actions`: (optional) actions to perform on the state after the numerical
  integration state update.

# Returns
- `X`: the integrated trajectory (final point, or full).
"""
function rk4(f::T.Func,
             x0::RealVector,
             tf::RealValue,
             h::RealValue;
             full::Bool=false,
             actions::T.SpecialIntegrationActions=
                 T.SpecialIntegrationActions(undef, 0))::Union{
                     RealVector, RealMatrix}
    # Define the scaled dynamics and time step
    F = (τ, x) -> tf*f(tf*τ, x)
    h /= tf

    # Integrate
    X = rk4_generic(F, x0; h=h, full=full, actions=actions)

    return X
end

"""
    rk4_core_step(f, x, t, tp)

Update the state using a single Runge-Kutta integration update.

# Arguments
- `f`: the dynamics function.
- `x`: the current state.
- `t`: the current time.
- `tp`: the next time.

# Returns
- `xp`: the next state.
"""
function rk4_core_step(f::T.Func,
                       x::RealVector,
                       t::RealValue,
                       tp::RealValue)::RealVector

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

"""
    rk4_generic(f, x0[; tspan, h, full, actions])

Classic Runge-Kutta integration. Interate a system of ordinary differential
equations over a time interval. The `tspan` and `h` arguments are mutually
exclusive: one and exactly one of them must be provided.

# Arguments
- `f`: the dynamics function, with call signature f(t, x) where t is the
  current time and x is the current state.
- `x0`: the initial condition.
- `tspan`: (optional) the time grid over which to integrate.
- `h`: (optional) the time step of a uniform time grid over which to
  integrate. If provided, a [0, 1] integration interval is assumed. If the time
  step does not split the [0, 1] interval into an integral number of temporal
  nodes, then the largest time step <=h is used.
- `full`: (optional) whether to return the complete trajectory (if false,
  return just the endpoint).
- `actions`: (optional) actions to perform on the state after the numerical
  integration state update.

# Returns
- `X`: the integration result (final point, or full trajectory, depending on
  the argument `full`).
"""
function rk4_generic(
    f::T.Func,
    x0::RealVector;
    tspan::Union{RealVector, Nothing}=nothing,
    h::Union{RealValue, Nothing}=nothing,
    full::Bool=false,
    actions::T.SpecialIntegrationActions=
        T.SpecialIntegrationActions(undef, 0))::Union{
            RealVector, RealMatrix}

    # Check that one and only one of the arguments tspan and h is passed
    if !xor(isnothing(tspan), isnothing(h))
        err = ArgumentError(string("rk4 accepts one and only one",
                                   " of tspan or h."))
        throw(err)
    end

    if isnothing(tspan)
        # Construct a uniform tspan such that the time step is the largest
        # possible value <=h
        N = ceil(Int, 1.0+1.0/h)
        tspan = LinRange(0.0, 1.0, N)
    end

    # Initialize the data
    n = length(x0)
    N = length(tspan)
    if full
        X = RealMatrix(undef, n, N)
        X[:, 1] = copy(x0)
    else
        X = copy(x0)
    end

    # Integrate the function
    for k = 2:N
        if full
            X[:, k] = rk4_core_step(f, X[:, k-1], tspan[k-1], tspan[k])
            # Update actions
            for act in actions
                X[act[1], k] = act[2](X[act[1], k])
            end
        else
            X = rk4_core_step(f, X, tspan[k-1], tspan[k])
            # Update actions
            for act in actions
                X[act[1]] = act[2](X[act[1]])
            end
        end
    end

    return X
end

"""
    expm_diff(A, DA[, c][; resol])

Compute the derivative of the matrix exponential [1]:

```math
\\frac{d}{d\\lambda} e^{c A} = \\int_0^c e^{xA} \\frac{dA}{d\\lambda}
 e^{(c-x)A}dx.
```

References:

[1] R. M. Wilcox (1967). "Exponential Operators and Parameter Differentiation
in Quantum Physics". Journal of Mathematical Physics. 8 (4):
962–982. doi:10.1063/1.1705306

# Arguments
- `A`: the matrix, which depends on a parameter ``\\lambda``,
  i.e. ``A(\\lambda)``.
- `DA`: the derivative of `A` with respect the parameter, i.e. ``dA/d\\lambda``.
- `c`: (optional) coefficient multiplying `A` in the matrix exponential, see
  the formula in the above description. Defaults to 1.

# Keywords
- `resol`: (optional) how many nodes to use of the integration grid.

# Returns
- `DecA`: the value ``\\frac{d}{d\\lambda} e^{c A}``.
"""
function expm_diff(A::RealMatrix, DA::RealMatrix, c::RealValue=1.0;
                   resol::Int=1000)::RealMatrix
    integrand = (t, _) -> begin
        dXdt = exp(t*A)*DA*exp((c-t)*A)
        dXdt = vec(dXdt)
        return dXdt
    end
    integ_grid = collect(LinRange(0, c, resol))
    DecA = reshape(rk4(integrand, zeros(length(A)), integ_grid), size(A))
    return DecA
end

"""
    trapz(f, grid)

Integrate a discrete signal using trapezoid rule.

# Arguments
- `f`: the function values on a discrete grid.
- `grid`: the discrete grid.

# Returns
- `F`: the numerical integral of the f signal.
"""
function trapz(f::AbstractVector, grid::RealVector)
    N = length(grid)
    F = 0.0
    for k = 1:N-1
        δ = grid[k+1]-grid[k]
        F += 0.5*δ*(f[k+1]+f[k])
    end
    return F
end

"""
    ∇trapz(grid)

Compute the gradient of the `trapz` function above with respect to its input
argument vector `f`.

# Arguments
- `grid`: the discrete grid.

# Returns
- `∇F`: the gradient of `trapz(F, grid)`.
"""
function ∇trapz(grid::RealVector)::RealVector
    N = length(grid)
    ∇F = zeros(N)
    for k = 1:N-1
        δ = grid[k+1]-grid[k]
        ∇F[k+1] += 0.5*δ
        ∇F[k] += 0.5*δ
    end
    return ∇F
end

"""
    logsumexp(f[, ∇f, ∇²f][; t])

Compute log-sum-exp function value and its gradient. Numerically stable
implementation based on [1]. The log-sum-exp function is:

```julia
L(x) = 1/t*log( ∑_i^{m} exp(t*f_i(x)) ).
```

As t increases, this becomes an increasingly accurate approximation of `max_i
f_i(x)`.

References:

[1] https://leimao.github.io/blog/LogSumExp/

# Arguments
- `f`: the function in the exponent.
- `∇f`: (optional) the gradient of the function.
- `∇²f`: (optional) the Hessian of the function.

# Keywords
- `t`: (optional) the homotopy parameter.

# Returns
- `L`: the log-sum-exp function value.
- `∇L`: the log-sum-exp function gradient. Only returned in ∇f provided.
- `∇²L`: the log-sum-exp function Hessian. Only returned in ∇²f provided.
"""
function logsumexp(
    f::RealVector,
    ∇f::Optional{Vector}=nothing,
    ∇²f::Optional{Vector}=nothing;
    t::RealValue=1.0)::Union{
        Tuple{RealValue, RealVector, RealMatrix},
        Tuple{RealValue, RealVector},
        RealValue}

    n = length(f)
    compute_gradient = !isnothing(∇f)
    compute_hessian = !isnothing(∇²f)

    # Value
    a = maximum(t*f)
    E = sum(exp(t*f[i]-a) for i=1:n)
    logsumexp = a+log(E)
    L = logsumexp/t

    if compute_gradient
        Ei = [exp(t*f[i]-a)/E for i=1:n]
        ∇L = sum(∇f[i]*Ei[i] for i=1:n)
        if compute_hessian
            ∇a = t*∇f[argmax(f)]
            ∇²L = sum((∇²f[i]+(∇f[i]-∇L)*(t*∇f[i]-∇a)')*Ei[i] for i=1:n)
            return L, ∇L, ∇²L
        end
        return L, ∇L
    end
    return L
end

"""
    sigmoid(value[, gradient, hessian][; κ])

Compute sigmoid function value. The sigmoid represents a logical "OR"
combination of its argument, in the sense that its value tends to one as any
argument becomes more positive.

# Arguments
- `value`: the function in the exponent.
- `gradient`: (optional) a list of gradients for each element of `value`.
- `hessian`: (optional) a list of Hessians for each element of `value`.

# Keywords
- `κ`: (optional) the sharpness parameter.

# Returns
- `σ`: the sigmoid function value (and its gradients and Hessians, depending on
  whether `gradient` and `hessian` are provided).
"""
function sigmoid(value::RealVector,
                 gradient::Optional{Vector}=nothing,
                 hessian::Optional{Vector}=nothing;
                 κ::RealValue=1.0)::Union{
                     Tuple{RealValue, RealVector, RealMatrix},
                     Tuple{RealValue, RealVector},
                     RealValue}

    compute_gradient = !isnothing(gradient)
    compute_hessian = !isnothing(hessian)

    L = logsumexp(value, gradient, hessian, t=κ)

    if compute_hessian
        L, ∇L, ∇²L = L
    elseif compute_gradient
        L, ∇L = L
    end

    σ = 1-1/(1+exp(κ*L))
    if compute_gradient
        c = exp(κ*L+2*log(1-σ))
        ∇σ = κ*c*∇L
        if compute_hessian
            ∇²σ = κ*(1-2*σ)*∇L*∇σ'+κ*c*∇²L
            return σ, ∇σ, ∇²σ
        end
        return σ, ∇σ
    end
    return σ
end

"""
    indicator(value[, gradient, hessian][; κ, match])

Sigmoid function that is matched with the exact value (i.e. the value that
would be obtained for `κ=Inf`) at the value `match` of `value`.

# Arguments
- `value`: vector value. The exact indicator returns one if any element of
  `value` is positive.
- `gradient`: (optional) a list of gradients for each element of `value`.
- `hessian`: (optional) a list of Hessians for each element of `value`.

# Keywords
- `κ`: (optional) sigmoid sharpness parameter.
- `match`: (optional) critical value of `value` at which to match the smoothed
  indicator value to the exact value.

# Returns
- `σ`: the sigmoid function value (and its gradients and Hessians, depending on
  whether `gradient` and `hessian` are provided).
"""
function indicator(value::RealVector,
                   gradient::Optional{Vector}=nothing,
                   hessian::Optional{Vector}=nothing;
                   κ::RealValue=1.0,
                   match::Optional{RealVector}=nothing)::Union{
                       Tuple{RealValue, RealVector, RealMatrix},
                       Tuple{RealValue, RealVector},
                       RealValue}

    compute_gradient = !isnothing(gradient)
    match_value = !isnothing(match)

    # Output value matching
    Δσ = 0
    if match_value
        offset = sigmoid(match; κ=κ)
        Δσ = 1-offset
    end

    # Compute sigmoid
    σ = sigmoid(value, gradient, hessian, κ=κ)
    if compute_gradient
        σ = (σ[1]+Δσ, σ[2:end]...)
        return σ
    end
    return σ+Δσ
end

"""
    or(predicates[, gradient, hessian][; κ, match, normalize])

A smooth logical OR. By passing the `normalize`, the function can be made scale
invariant, i.e. the shape of the OR function will not change due to uniform
scaling of the predicates.

# Arguments
- `predicates`: the predicates to be composed with logical "OR". Each predicate
  element is taken as "true" if it is positive.
- `gradient`: (optional) a list of gradients for each predicate.
- `hessian`: (optional) a list of Hessians for each predicate.

# Keywords
- `κ`: (optional) sigmoid sharpness parameter.
- `match`: (optional) value of the predicates at which to match the smooth or
  function to its exact value, via y-shifting.
- `normalize`: (optional) normalization value to divide the predicates by
  (e.g., their expected maximum value).

# Returns
- `OR`: the smooth OR value, together with its gradient and Hessian (if the
  gradients and hessians of the predicates are provided)
"""
function or(predicates::RealVector,
            gradient::Optional{Vector}=nothing,
            hessian::Optional{Vector}=nothing;
            κ::RealValue=1.0,
            match::Optional{Union{RealValue, RealVector}}=nothing,
            normalize::RealValue=1.0)::Union{
                Tuple{RealValue, RealVector, RealMatrix},
                Tuple{RealValue, RealVector},
                RealValue}

    @assert normalize>0
    @assert isnothing(match) || any(match.>0)

    scale = p -> p/normalize
    match = isnothing(match) ? match : match./normalize
    if !isnothing(match) && !(match isa AbstractArray)
        match = [match]
    end

    compute_gradient = !isnothing(gradient)
    compute_hessian = !isnothing(hessian)

    predicates = collect(map(scale, predicates))
    if compute_gradient
        gradient = collect(map(scale, gradient))
    end
    if compute_hessian
        hessian = collect(map(scale, hessian))
    end

    OR = indicator(predicates, gradient, hessian,
                   κ=κ, match=match)

    return OR
end

"""
    squeeze(A)

Remove all length-1 dimensions from the array `A`.

# Arguments
- `A`: the original array.

# Returns
- `Ar`: the array with all dimensions of length 1 removed..
"""
function squeeze(A::AbstractArray)::AbstractArray
    singleton_dims = tuple((d for d in 1:ndims(A) if size(A, d) == 1)...)
    Ar = dropdims(A, dims=singleton_dims)
    return Ar
end

"""
    convert_units(x, orig, new)

Convert the units of `x`.

# Arguments
- `x`: the value in the original units.
- `orig`: the original units.
- `new`: the new units.

# Returns
- `y`: the value in the new units.
"""
function convert_units(x::RealValue, orig::Symbol, new::Symbol)::RealValue

    convert_func = Symbol("_"*string(orig)*"2"*string(new))
    y = eval( :( $convert_func($x) ) )

    return y
end

# Available converters
_deg2rad(x) = deg2rad(x)        # degrees to radians
_in2m(x) = x*0.0254             # inches to meters
_ft2m(x) = x*0.3048             # feet to meters
_ftps2mps(x) = x*0.3048         # feet per second to meters per second
_ft2slug2m2kg(x) = x*1.35581795 # slug*ft^2 to kg*m^2
_lb2kg(x) = x*0.453592          # lb to kg
_lbf2N(x) = x*4.448222          # lbf to N

"""
    homtransf(q, r)

Build a homogeneous transformation 4x4 matrix between two matrix. The
quaternion `q` specifies the passive rotation (i.e. coordinate change) from
frame `B` to frame `A`. The vector `r` specifies the position of the `B` frame
origin with respect to the `A` frame origin, expressed in the `A` frame. The
function outputs a matrix `T` such that, given a homogeneous vector `x` in the
`B` frame, `y=T*x` yields a homogeneous vector `y` in the `A` frame.

# Arguments
- `q`: the quaternion specifying the rotation component.
- `r`: the vector specifying the displacement component.

# Returns
- `T`: the homogeneous transformation matrix.
"""
function homtransf(q::Quaternion, r::RealVector)::RealMatrix
    R = dcm(q)
    T = [R r; zeros(1, 3) 1]
    return T
end

""" Aliases for calling `homtrans` with pure rotation or displacement. """
homtransf(q::Quaternion) = homtransf(q, zeros(3))

"""
    homtransf([r][;, roll, pitch, yaw, deg])

Alias for calling `homtransf` with roll/pitch/yaw angles. See the documentation
of `homtransf` for further information, as this function simply converts its
inputs to the format ingested by the core `homtransf` function above. The angle
sequence is YAW-PITCH-ROLL according to the intrinsic Tait-Bryan
convention. See the `rpy` function in `quaternion.jl` for more information.

# Arguments
- `r`: (optional) the displacement.
- `roll`: (optional) the roll angle (about `+x''`).
- `pitch`: (optional) the pitch angle (about `+y'`).
- `yaw`: (optional) the yaw angle (about `+z`).
- `deg`: (optional) specify the angles in degrees if true.

# Returns
- `T`: the homogeneous transformation matrix.
"""
function homtransf(r::RealVector=zeros(3);
                   roll::RealValue=0,
                   pitch::RealValue=0,
                   yaw::RealValue=0,
                   deg::Bool=true)::RealMatrix
    if deg
        roll = deg2rad(roll)
        pitch = deg2rad(pitch)
        yaw = deg2rad(yaw)
    end
    q_roll = Quaternion(roll, [1; 0; 0])
    q_pitch = Quaternion(pitch, [0; 1; 0])
    q_yaw = Quaternion(yaw, [0; 0; 1])
    q = q_yaw*q_pitch*q_roll
    return homtransf(q, r)
end

""" Convenience methods to extract rotation and displacement part of
homogeneous transformation matrix. """
homdisp(T::RealMatrix)::RealVector = T[1:3, 4]
homrot(T::RealMatrix)::RealMatrix = T[1:3, 1:3]

"""
    hominv(T)

Compute the inverse homogeneous transformation matrix. If `y=T*x`, this
function returns `inv(T)` such that `x=inv(T)*y`.

# Arguments
- `T`: the "forward" homogeneous transformation.

# Returns
- `iT`: the "reverse" homogeneous transformation.
"""
function hominv(T::RealMatrix)::RealMatrix
    R = homrot(T)
    v = homdisp(T)
    iR = R' # Inverse rotation
    iT = [iR -iR*v; zeros(3)' 1]
    return iT
end

""" Print a heading for the test. """
test_heading(description) = printstyled(
    @sprintf("%s\n", description),
    color=:blue, bold=true)

"""
    make_indent(io)

Make an indent string prefix.

# Arguments
- `io`: stream object.

# Returns
- `indeint`: the indent string.
"""
function make_indent(io::IO)::String
    indent = " "^get(io, :indent, 0)
    return indent
end

"""
    printf_prefixed(io, prefix, fmt, data...)

Same as `@printf` macro, except insert a prefix at the start of the string.

# Arguments
- `io`: stream object.
- `prefix`: a string to be prefixed to the beginning.
- `fmt`: the format specified for the rest of the string.
- `data...`: the data to be printed according to the format specifier.
"""
macro preprintf(io, prefix, fmt, data...)
    new_fmt = "%s"*fmt
    quote
        @printf($(esc(io)), $new_fmt, $(esc(prefix)), $(esc.(data)...))
    end
end # macro
