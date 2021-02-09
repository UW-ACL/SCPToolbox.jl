#=Helper functions used throughout the code=#

include("types.jl")

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
    v0::T_RealVector, vf::T_RealVector, N::T_Int)::T_RealMatrix
    nv = length(v0)
    v = zeros(nv,N)
    for k = 1:nv
	v[k,:] = range(v0[k],stop=vf[k],length=N)
    end
    return v
end

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
function linterp(t::Float64, f_cps::T_RealMatrix,
                 t_grid::T_RealVector)::T_RealVector
    k = get_interval(t, t_grid)
    c = (t_grid[k+1]-t)/(t_grid[k+1]-t_grid[k])
    f_t = c*f_cps[:,k] + (1-c)*f_cps[:,k+1]
    return f_t
end

#= Compute grid bin.

Get which grid interval a real number belongs to.

Args:
    x: the real number.
    grid: the discrete grid on the real line.

Returns:
    k: the interval of the grid the the real number is in.=#
function get_interval(x::T_Real, grid::T_RealVector)::T_Int
    k = sum(x.>grid)
    if k==0
	k = 1
    end
    return k
end

#= Classic Runge-Kutta integration.

Interate a system of ordinary differential equations over a time interval, and
return the final result.

Args:
    x0: the initial condition.
    f: the ODE.
    tspan: the time interval over which to integrate.

Returns:
    x: the integration result at the end of the time interval.=#
function rk4(f, x0::T_RealVector, tspan::T_RealVector)::T_RealVector
    x = copy(x0)
    n = length(x)
    N = length(tspan)
    for k = 1:N-1
	tk = tspan[k]
  	tkp1 = tspan[k+1]
	h = tkp1-tk

	k1 = f(tk,x)
	k2 = f(tk+h/2,x+h/2*k1)
	k3 = f(tk+h/2,x+h/2*k2)
	k4 = f(tk+h,x+h*k3)

	x += h/6*(k1+2*k2+2*k3+k4)
    end
    return x
end

#= Integrate a discrete signal using trapezoid rule.

Args:
    f: the function values on a discrete grid.
    grid: the discrete grid.

Returns:
    F: the numerical integral of the f signal. =#
function trapz(f, grid::T_RealVector)
    N = length(grid)
    F = 0.0
    for k = 1:N-1
        δ = grid[k+1]-grid[k]
        F += 0.5*δ*(f[k+1]+f[k])
    end
    return F
end
