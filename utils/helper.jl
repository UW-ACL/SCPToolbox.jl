#= Helper functions used throughout the code. =#

using LinearAlgebra

include("types.jl")

# ..:: Public macros ::..

#= Get discrete-time trajectory values at time step k or step range {k,...,l}.

The trajectory can be represented by an array of any dimension. We assume that
the time axis is contained in the **last** dimension.

The macros below are overloaded to correspond to the following possible inputs.

Args:
    traj: discrete-time trajectory representation of the type Array.
    k: (optional) the time step at which to get the trajectory value. Defaults
        to the current value of k in the workspace.
    l: (optional) the final time step, in case you want to get a rank of values
        from time step k to time step l.

Returns:
    The trajectory value at time step k or range {k,...,l}. =#
macro k(traj)
    :( $(esc(traj))[fill(:, ndims($(esc(traj)))-1)..., $(esc(:(k)))] )
end

macro k(traj, k)
    :( $(esc(traj))[fill(:, ndims($(esc(traj)))-1)..., $(esc(k))] )
end

macro k(traj, k, l)
    :( $(esc(traj))[fill(:, ndims($(esc(traj)))-1)..., $(esc(k)):$(esc(l))] )
end

#= Convenience method to get trajectory value at time k+1.

Args:
    traj: discrete-time trajectory representation of the type Array.

Returns:
    The trajectory value at time step k+1. =#
macro kp1(traj)
    :( @k($(esc(traj)), $(esc(:(k)))+1) )
end

#= Convenience method to get trajectory value at time k-1.

Args:
    traj: discrete-time trajectory representation of the type Array.

Returns:
    The trajectory value at time step k-1. =#
macro km1(traj)
    :( @k($(esc(traj)), $(esc(:(k)))-1) )
end

#= Convenience method to get trajectory value at the initial time.

Args:
    traj: discrete-time trajectory representation of the type Array.

Returns:
    The trajectory initial value. =#
macro first(traj)
    :( @k($(esc(traj)), 1) )
end

#= Convenience method to get trajectory value at the final time.

Args:
    traj: discrete-time trajectory representation of the type Array.

Returns:
    The trajectory final value. =#
macro last(traj)
    :( @k($(esc(traj)), size($(esc(traj)))[end]) )
end

# ..:: Public methods ::..

#= Linear interpolation on a grid.

Linearly interpolate a discrete function on a time grid. In other words, get
the function value assuming that it is a continuous and piecewise affine
function.

Args:
    t: the time at which to get the function value.
    f_cps: the control points of the function, stored as columns of a matrix.
    t_grid: the discrete time nodes.

Returns:
    f_t: the function value at time t.=#
function linterp(t::T_Real,
                 f_cps::T_RealArray,
                 t_grid::T_RealVector)::T_RealArray
    t = max(t_grid[1], min(t_grid[end], t)) # Saturate to time grid
    k = _helper__get_interval(t, t_grid)
    c = (@kp1(t_grid)-t)/(@kp1(t_grid)-@k(t_grid))
    f_t = c*@k(f_cps) + (1-c)*@kp1(f_cps)
    return f_t
end

#= Straight-line interpolation between two points.

Compute a straight-line interpolation between an initial and a final vector, on
a grid of N points.

Args:
    v0: starting vector.
    vf: ending vector.
    N: number of vectors in between (including endpoints).

Returns:
    v: the resulting interpolation (a matrix, k-th column is the k-th vector,
       k=1,...,N).=#
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

#= Get the value of a continuous-time trajectory at time t.

Args:
    traj: the trajectory.
    t: the evaluation time.

Returns:
    x: the trajectory value at time t. =#
function sample(traj::ContinuousTimeTrajectory,
                t::T_Real)::T_RealArray

    if traj.interp==:linear
        x = linterp(t, traj.x, traj.t)
    end

    return x
end

#= Classic Runge-Kutta integration over a provided time grid.

Args:
    f: the dynamics function.
    x0: the initial condition.
    tspan: the discrete time grid over which to integrate.
    full: (optional) whether to return the full trajectory, or just
        the final point.

Returns:
    X: the integrated trajectory (final point, or full). =#
function rk4(f::Function,
             x0::T_RealVector,
             tspan::T_RealVector;
             full::T_Bool=false)::Union{T_RealVector,
                                        T_RealMatrix}
    X = _helper__rk4_generic(f, x0; tspan=tspan, full=full)
    return X
end

#= Classic Runge-Kutta integration given a final time and time step.

Args:
    f: the dynamics function.
    x0: the initial condition.
    tf: the final time.
    h: the discrete time step.
    full: (optional) whether to return the full trajectory, or just
        the final point.

Returns:
    X: the integrated trajectory (final point, or full). =#
function rk4(f::Function,
             x0::T_RealVector,
             tf::T_Real,
             h::T_Real;
             full::T_Bool=false)::Union{T_RealVector,
                                        T_RealMatrix}
    # Define the scaled dynamics and time step
    F = (τ::T_Real, x::T_RealVector) -> tf*f(tf*τ, x)
    h /= tf

    # Integrate
    X = _helper__rk4_generic(F, x0; h=h, full=full)

    return X
end

#= Integrate a discrete signal using trapezoid rule.

Args:
    f: the function values on a discrete grid.
    grid: the discrete grid.

Returns:
    F: the numerical integral of the f signal. =#
function trapz(f::AbstractVector, grid::T_RealVector)
    N = length(grid)
    F = 0.0
    for k = 1:N-1
        δ = @kp1(grid)-@k(grid)
        F += 0.5*δ*(@kp1(f)+@k(f))
    end
    return F
end

#= Project an ellipsoid onto a subset of its axes.

Args:
    H: ellipsoid shape matrix.
    c: ellipsoid center.
    ax: array of the axes onto which to project.

Returns:
    H_prj: projected ellipsoid shape matrix.
    c_prj: projected ellipsoid center. =#
function project(H::T_RealMatrix,
                 c::T_RealVector,
                 ax::T_IntVector)::Tuple{T_RealMatrix,
                                         T_RealVector}
    # Parameters
    n = size(H, 1)
    m = length(ax)

    # Projection matrix onto lower-dimensional space
    P = zeros(m, n)
    for i=1:m
        P[i, ax[i]] = 1.0
    end

    # Do the projection
    Hb = P*H
    F = svd(Hb)
    H_prj = F.U*diagm(F.S)
    c_prj = P*c

    return H_prj, c_prj
end

# ..:: Private methods ::..

#= Compute grid bin.

Get which grid interval a real number belongs to.

Args:
    x: the real number.
    grid: the discrete grid on the real line.

Returns:
    k: the interval of the grid the the real number is in.=#
function _helper__get_interval(x::T_Real, grid::T_RealVector)::T_Int
    k = sum(x.>grid)
    if k==0
	k = 1
    end
    return k
end

#= Update the state using a single Runge-Kutta integration update.

Args:
    f: the dynamics function.
    x: the current state.
    t: the current time.
    tp: the next time.

Returns:
    xp: the next state. =#
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

#= Classic Runge-Kutta integration.

Interate a system of ordinary differential equations over a time interval. The
`tspan` and `h` arguments are mutually exclusive: one and exactly one of them
must be provided.

Args:
    f: the dynamics function, with call signature f(t, x) where t is the
        current time and x is the current state.
    x0: the initial condition.
    tspan: (optional) the time grid over which to integrate.
    h: (optional) the time step of a uniform time grid over which to integrate.
        If provided, a [0, 1] integration interval is assumed. If the time step
        does not split the [0, 1] interval into an integral number of temporal
        nodes, then the largest time step <=h is used.
    full: (optional) whether to return the complete trajectory (if false, return
        just the endpoint).

Returns:
    X: the integration result (final point, or full trajectory, depending on the
        argument `full`).=#
function _helper__rk4_generic(f::Function,
                              x0::T_RealVector;
                              tspan::Union{T_RealVector, Nothing}=nothing,
                              h::Union{T_Real, Nothing}=nothing,
                              full::T_Bool=false)::Union{T_RealVector,
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
        else
            X = _helper__rk4_core_step(
                f, X, @km1(tspan), @k(tspan))
        end
    end

    return X
end
