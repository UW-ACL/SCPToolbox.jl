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

if isdefined(@__MODULE__, :LanguageServer)
    include("Utils.jl")
    import .Utils.Types
end

const T = Types
const RealValue = T.RealTypes
const RealArray = T.RealArray
const RealVector = T.RealVector
const RealMatrix = T.RealMatrix

export skew, get_interval, linterp, zohinterp, diracinterp,
    straightline_interpolate, rk4, trapz, ∇trapz, logsumexp, or, squeeze,
    convert_units

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
end # function

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
end # function

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
    k = get_interval(t, t_grid) #noinfo
    c = (t_grid[k+1]-t)/(t_grid[k+1]-t_grid[k])
    dimfill = fill(:, ndims(f_cps)-1)
    f_t = c*f_cps[dimfill..., k]+(1-c)*f_cps[dimfill..., k+1]
    return f_t
end # function

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
end # function

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
end # function

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
end # function

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
end # function

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
end # function

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
end # function

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
end # function

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
end # function

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
end # function

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
end # function

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
    f::RealVector,
    ∇f::Union{Nothing, Vector{Vector{Float64}}}=nothing;
    t::RealValue=1.0)::Union{
        Tuple{RealValue, RealVector},
        RealValue}

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
end # function

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
function sigmoid(f::RealVector,
                 ∇f::Union{Nothing, Vector{Vector{Float64}}}=nothing;
                 κ::RealValue=1.0)::Union{
                     Tuple{RealValue, RealVector},
                     RealValue}

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
end # function

"""
    indicator(f[, ∇f][; κ1, κ2])

Like a sigmoid, except that for very small κ1 values the function is
approximately equal to one everywhere, and not 0.5 as is the case for a
sigmoid. This, low κ1 values are associated with "always in the set", where the
set is defined as any time either element of f is nonnegative.

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
function indicator(f::RealVector,
                   ∇f::Union{Nothing, Vector{Vector{Float64}}}=nothing;
                   κ1::RealValue=1.0,
                   κ2::RealValue=1.0)::Union{
                       Tuple{RealValue, RealVector},
                       RealValue}

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
end # function

"""
    or(p1[, p2, ..., pN][; κ1, κ2, minval, maxval])

A smooth logical OR. By passing the `minval` and `maxval`, the function can be
made scale invariant, i.e. the shape of the OR function will not change due to
uniform scaling of the predicates.

# Arguments
- `predicates`: (optional) the predicates to be composed as logical
  "OR". Provide either a list of reals `f` or a list of (real, vector) tuples
  `(f, ∇f)`.

# Keywords
- `κ1`: (optional) sigmoid sharpness parameter.
- `κ2`: (optional) normalization sharpness parameter.
- `minval`: (optional) minimum value of predicate.
- `maxval`: (optional) maximum value of predicate.

# Returns
- `OR`: the smooth OR value (1≡true).
- `∇OR`: the smooth OR gradient. Only returned in `∇f` provided.
"""
function or(predicates...;
            κ1::RealValue=1.0,
            κ2::RealValue=1.0,
            minval::RealValue=-1.0,
            maxval::RealValue=1.0)::Union{
                Tuple{RealValue, RealVector},
                RealValue}

    c = (maxval+minval)/2
    rng = (maxval-minval)/2
    scale = (p) -> (p-c)/rng
    if typeof(predicates[1])<:Tuple
        ∇scale = (∇p) -> ∇p/(maxval-minval)
        f = RealVector([scale(p[1]) for p in predicates])
        ∇f = [∇scale(p[2]) for p in predicates]
        OR, ∇OR = indicator(f, ∇f; κ1=κ1, κ2=κ2)
        return OR, ∇OR
    else
        f = RealVector([scale(p) for p in predicates])
        OR = indicator(f; κ1=κ1, κ2=κ2)
        return OR
    end

end # function

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
end # function

function convert_units(x::RealValue, converter::Symbol)::RealValue
    convert_func = Symbol("_"*string(converter))
    y = eval(Expr(:call, convert_func, x))
    return y
end # function

_in2m(x) = x*0.0254 # inches to meters
_ft2slug2m2kg(x) = x*1.35581795 # slug*ft^2 to kg*m^2
_lb2kg(x) = x*0.453592 # lb to kg
_lbf2N(x) = x*4.448222 # lbf to N
