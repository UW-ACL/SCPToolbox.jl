#= Lossless convexification double integrator problem solvers.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

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
using ECOS
using Printf

"""
    solve_lcvx(mdl)

Numerical solution of the double integrator trajectory generation problem, via
lossless convexification.

# Arguments
- `mdl`: the double integrator model.

# Returns
- `sol`: the optimal solution.

# Throws
- `ErrorException` if the solver does not return `OPTIMAL` status.
"""
function solve_lcvx(mdl::DoubleIntegratorParameters)::Solution

    # Parameters
    T = mdl.T
    N = mdl.N
    A = mdl.A
    Bm = mdl.Bm
    Bp = mdl.Bp
    w = mdl.w
    s = mdl.s
    dt = T / (N - 1)

    ocp = ConicProgram(
        nothing;
        solver = ECOS,
        solver_options = Dict("verbose" => 0),
    )

    x = @new_variable(ocp, (2, N), "x")
    u = @new_variable(ocp, (1, N), "u")
    σ = @new_variable(ocp, (1, N), "σ")
    σ² = @new_variable(ocp, (1, N), "σ²")

    @add_constraint(ocp, ZERO, "initial_condition", (x[:, 1],), begin
        local x0 = arg[1]
        x0
    end)

    @add_constraint(ocp, ZERO, "final_condition", (x[:, end],), begin
        local xf = arg[1]
        xf - [s; 0]
    end)

    for k = 1:N
        @add_constraint(ocp, NONPOS, "sigma_ub", (σ[k],), begin
            local σk = arg[1]
            σk - 2
        end)
        @add_constraint(ocp, NONPOS, "sigma_lb", (σ[k],), begin
            local σk = arg[1]
            1 - σk
        end)
        @add_constraint(ocp, L1, "lcvx_equality", (u[k], σ[k]), begin
            local uk, σk = arg
            vcat(σk, uk)
        end)
        @add_constraint(ocp, GEOM, "slack_squared", (σ[k], σ²[k]), begin
            local σk, σ²k = arg
            vcat(σk, σ²k, 1)
        end)
        if k < N
            @add_constraint(
                ocp,
                ZERO,
                "dynamics",
                (x[:, k+1], x[:, k], u[:, k], u[:, k+1]),
                begin
                    local xn, x, u, un = arg
                    xn - (A * x + Bm .* u + Bp .* un + w)
                end
            )
        end
    end

    @add_cost(ocp, (σ²,), begin
        local σ² = arg[1]
        sum(σ²) * dt
    end)

    status = solve!(ocp)

    if status != MOI.OPTIMAL
        msg = @sprintf("Numerical solution error (%s)", status)
        throw(ErrorException(msg))
    end

    x = value(x)
    u = value(u)

    t = collect(LinRange(0, T, N))
    sol = Solution(t, x, u)
    cost = objective_value(ocp)

    return sol
end

"""
    solve_mp(mdl)

Analytical solution of the double integrator trajectory generation problem, via
a shooting method that solves for the necessary conditions of optimality
obtained from Pontryagin's maximum principle.

# Arguments
- `mdl`: the double integrator model.

# Returns
- `sol`: the optimal solution.

# Throws
- `ErrorException` if the shooting method fails to satisfy the terminal
  condition to within a specified tolerance.
"""
function solve_mp(mdl::DoubleIntegratorParameters)::Solution

    # Parameters
    f = mdl.f
    T = mdl.T
    s = mdl.s
    runsim = (c, ts) -> mp_sim(f, T, s, c, ts)
    N_grid = 25
    tol_err = 1e-2
    max_iter = 10
    if mdl.choice == 1
        c_range = [-3; -1]
        ts_range = [4.5; 5.5]
    else
        c_range = [-1.5; -0.5]
        ts_range = [6.5; 7.5]
    end

    # Find optimal solution's (c, ts) parameters via shooting method and
    # iterative grid search
    c_grid = LinRange(c_range[1], c_range[2], N_grid)
    ts_grid = LinRange(ts_range[1], ts_range[2], N_grid)
    c_grid = c_grid' .* ones(N_grid)
    ts_grid = ones(N_grid)' .* ts_grid
    err_grid = fill(NaN, N_grid, N_grid)
    c, ts = nothing, nothing
    iter = 1

    while true
        # Evaluate errors for current grid of (c, ts)
        for j = 1:N_grid
            for i = 1:N_grid
                out = runsim(c_grid[i, j], ts_grid[i, j])
                err_grid[i, j] = out[:err]
            end
        end

        # Get current minimum error
        err_grid_padded = err_grid[2:N_grid-1, 2:N_grid-1]
        c_grid_padded = c_grid[2:N_grid-1, 2:N_grid-1]
        ts_grid_padded = ts_grid[2:N_grid-1, 2:N_grid-1]
        err_idx = argmin(err_grid_padded[:])
        err_min = err_grid_padded[err_idx]
        @printf("MP terminal error = %.3e\n", err_min)
        if err_min <= tol_err
            # Solution found
            c = c_grid_padded[err_idx]
            ts = ts_grid_padded[err_idx]
            break
        end

        # Set grid to neighborhood of current best (c, ts) values
        ij = CartesianIndices((N_grid - 2, N_grid - 2))[err_idx]
        i, j = ij[1], ij[2]
        i += 1 # convert padded->not padded row index
        j += 1 # convert padded->not padded column index
        c_grid = LinRange(c_grid[i, j-1], c_grid[i, j+1], N_grid)
        ts_grid = LinRange(ts_grid[i-1, j], ts_grid[i+1, j], N_grid)
        c_grid = c_grid' .* ones(N_grid)
        ts_grid = ones(N_grid)' .* ts_grid

        # Update iteration counter
        iter += 1
        if iter > max_iter
            throw(ErrorException("failed to find a solution"))
        end
    end

    # Store the analytical optimal solution
    out = runsim(c, ts)
    t_mp = out[:t]
    x_mp = out[:x]
    p = c * (out[:t] .- ts)
    u_mp = fill(NaN, 1, length(t_mp))
    for k = 1:length(t_mp)
        u_mp[k] = mp_input(p[k])
    end
    sol = Solution(t_mp, x_mp, u_mp)

    return sol
end

"""
    mp_input(p)

Optimal control according to the maximum principle.

# Arguments
- `p` : adjoint variable.

# Returns
- `u` : optimal control.
"""
function mp_input(p::Float64)::Float64

    if p > 4
        u = 2
    elseif p >= 2 && p <= 4
        u = p / 2
    elseif p >= 0 && p < 2
        u = 1
    elseif p >= -2 && p < 0
        u = -1
    elseif p >= -4 && p < -2
        u = p / 2
    else
        u = -2
    end

    return u
end

"""
    mp_sim(f, T, s, c, ts)

Simulate the dynamics using a guess for the adjoint variable trajectory.

# Arguments
- `f`: dynamics.
- `T`: terminal t_lcvx.
- `s`: terminal position.
- `c`: adjoint slope.
- `ts`: adjoint switch t_lcvx (t_lcvx at which adjoint=0).

# Returns
- `out` : information about this solution.
"""
function mp_sim(f::Function, T::Real, s::Real, c::Real, ts::Real)::Dict

    # Adjoint variable trajectory
    p = t -> c * (t - ts)

    # Time intervals between input switches
    t_crit = [ts + a / c for a in [4, 2, 0, -2, -4]]
    filter!(τ -> τ >= 0 && τ <= T, t_crit)
    pushfirst!(t_crit, 0)
    push!(t_crit, T)
    t_grid = [collect(LinRange(t_crit[i:i+1]..., 100)) for i = 1:length(t_crit)-1]

    # Integrate for each time interval
    x = Matrix[]
    for i = 1:length(t_grid)
        x0 = (i == 1) ? zeros(2) : x[i-1][:, end]
        push!(x, rk4((t, x) -> f(t, x, mp_input(p(t))), x0, t_grid[i]; full = true))
    end

    # Combine all into a single trajectory
    t_grid = vcat(t_grid...)
    x = hcat(x...)

    # Extract error at final time
    xf = x[:, end]
    err = norm(xf - [s; 0])

    out = Dict(:c => c, :ts => ts, :err => err, :t => t_grid, :x => x)

    return out
end
