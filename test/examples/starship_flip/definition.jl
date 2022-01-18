#= Starship landing flip maneuver problem definition.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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

using JuMP
using ECOS
using Printf

# ..:: Methods ::..

function define_problem!(pbm::TrajectoryProblem, algo::Symbol)::Nothing
    set_dims!(pbm)
    set_scale!(pbm)
    set_cost!(pbm)
    set_dynamics!(pbm)
    set_convex_constraints!(pbm)
    set_nonconvex_constraints!(pbm, algo)
    set_bcs!(pbm)

    set_guess!(pbm)

    return nothing
end

function set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 8, 3, 10)

    return nothing
end

function set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl
    veh = mdl.vehicle
    traj = mdl.traj

    advise! = problem_advise_scale!

    # States
    advise!(pbm, :state, veh.id_r[1], (-100.0, 100.0))
    advise!(pbm, :state, veh.id_r[2], (0.0, traj.r0[2]))
    advise!(pbm, :state, veh.id_v[1], (-10.0, 10.0))
    advise!(pbm, :state, veh.id_v[2], (traj.v0[2], 0.0))
    advise!(pbm, :state, veh.id_θ, (0.0, traj.θ0))
    advise!(pbm, :state, veh.id_ω, deg2rad.((-10.0, 10.0)))
    advise!(pbm, :state, veh.id_m, (veh.m - 1e3, veh.m))
    advise!(pbm, :state, veh.id_δd, (-veh.δ_max, veh.δ_max))
    # Inputs
    advise!(pbm, :input, veh.id_T, (veh.T_min1, veh.T_max3))
    advise!(pbm, :input, veh.id_δ, (-veh.δ_max, veh.δ_max))
    advise!(pbm, :input, veh.id_δdot, (-veh.δdot_max, veh.δdot_max))
    # Parameters
    advise!(pbm, :parameter, veh.id_t1, (0.0, traj.tf_max))
    advise!(pbm, :parameter, veh.id_t2, (0.0, traj.tf_max))
    for i = 1:pbm.nx
        advise!(pbm, :parameter, veh.id_xs[i], pbm.xrg[i])
    end

    return nothing
end

""" Compute the initial trajectory guess.

This uses a simple bang-bang control strategy for the flip maneuver. Once
Starship is upright, convex optimization is used to find the terminal descent
trajectory by approximatin Starship as a double-integrator (no attitude, no
aerodynamics).

Args:
* `N`: the number of discrete-time grid nodes.
* `pbm`: the trajectory problem structure.

Returns:
* `x_guess`: the state trajectory initial guess.
* `u_guess`: the input trajectory initial guess.
* `p_guess`: the parameter vector initial guess.
"""
function starship_initial_guess(
    N::Int,
    pbm::TrajectoryProblem,
)::Tuple{RealMatrix,RealMatrix,RealVector}

    @printf("Computing initial guess .")

    # Parameters
    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj
    env = pbm.mdl.env

    # Normalized time grid
    τ_grid = RealVector(LinRange(0.0, 1.0, N))
    id_phase1 = findall(τ_grid .<= traj.τs)
    id_phase2 = IntRange(id_phase1[end]:N)

    # Initialize empty trajectory guess
    x_guess = zeros(pbm.nx, N)
    u_guess = zeros(pbm.nu, N)

    ######################################################
    # Phase 1: flip ######################################
    ######################################################

    # Simple guess control strategy
    # Gimbal bang-bang drive θ0 to θs at min 3-engine thrust
    flip_ac = veh.lcg / veh.J * veh.T_min3 * sin(veh.δ_max)
    flip_ts = sqrt((traj.θ0 - traj.θs) / flip_ac)
    flip_ctrl = (t, pbm) -> begin
        veh = pbm.mdl.vehicle
        T = veh.T_min3
        ts = flip_ts
        if t <= ts
            δ = veh.δ_max
        elseif t > ts && t <= 2 * ts
            δ = -veh.δ_max
        else
            δ = 0.0
        end
        u = zeros(pbm.nu)
        u[veh.id_T] = T
        u[veh.id_δ] = δ
        return u
    end

    # Dynamics with guess control
    flip_f = (t, x, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        u = flip_ctrl(t, pbm)
        k = max(floor(Int, t / (N - 1)) + 1, N)
        p = zeros(pbm.np)
        p[veh.id_t1] = traj.τs
        p[veh.id_t2] = 1 - traj.τs
        dxdt = dynamics(t, k, x, u, p, pbm; no_aero_torques = true)
        return dxdt
    end

    # Initial condition
    x10 = zeros(pbm.nx)
    x10[veh.id_r] = traj.r0
    x10[veh.id_v] = traj.v0
    x10[veh.id_θ] = traj.θ0
    x10[veh.id_δd] = veh.δ_max

    # Propagate the flip dynamics under the guess control
    t_θcst = 10.0
    tf = 2 * flip_ts + t_θcst
    t = RealVector(LinRange(0.0, tf, 5000))
    x1 = rk4((t, x) -> flip_f(t, x, pbm), x10, t; full = true)

    # Find crossing of terminal vertical velocity
    vs = dot(traj.vs, env.ey)
    k_0x = findfirst(x1[veh.id_v, :]' * env.ey .>= vs)
    if isnothing(k_0x)
        msg = string("no terminal velocity crossing, ", "increase time of flight (t_θcst).")
        error = ArgumentError(msg)
        throw(error)
    end
    t = t[1:k_0x]
    t1 = t[end]
    x1 = x1[:, 1:k_0x]

    # Populate trajectory guess first phase
    τ2t = (τ) -> τ / traj.τs * t1
    x1c = ContinuousTimeTrajectory(t, x1, :linear)
    x_guess[:, id_phase1] = hcat([sample(x1c, τ2t(τ)) for τ in τ_grid[id_phase1]]...)
    u_guess[:, id_phase1] = hcat([flip_ctrl(τ2t(τ), pbm) for τ in τ_grid[id_phase1]]...)

    @printf(".")

    ######################################################
    # Phase 2: terminal descent ##########################
    ######################################################

    # Get the transition state
    xs = sample(x1c, τ2t(τ_grid[id_phase1[end]]))
    traj.hs = dot(xs[veh.id_r], env.ey)

    # Discrete time grid
    τ2 = τ_grid[id_phase2] .- τ_grid[id_phase2[1]]
    N2 = length(τ2)
    tdil = (t2) -> t2 / (1 - traj.τs) # Time dilation amount

    # State and control dims for simple system
    nx = 4
    nu = 2

    # LTI state space matrices
    A_lti = [zeros(2, 2) I(2); zeros(2, 4)]
    B_lti = [zeros(2, 2); I(2) / veh.m]
    r_lti = [zeros(2); env.g]

    # Matrix indices in concatenated vector
    idcs_A = (1:nx*nx)
    idcs_Bm = (1:nx*nu) .+ idcs_A[end]
    idcs_Bp = (1:nx*nu) .+ idcs_Bm[end]
    idcs_r = (1:nx) .+ idcs_Bp[end]

    # Concatenated time derivative for propagation
    derivs = (t, V, Δt, tdil) -> begin
        # Get current values
        Phi = reshape(V[idcs_A], (nx, nx))
        σm = (Δt - t) / Δt
        σp = t / Δt

        # Apply time dilation to integrate in absolute time
        _A = tdil * A_lti
        _B = tdil * B_lti
        _r = tdil * r_lti

        # Compute derivatives
        iPhi = Phi \ I(nx)
        dPhidt = _A * Phi
        dBmdt = iPhi * _B * σm
        dBpdt = iPhi * _B * σp
        drdt = iPhi * _r

        dVdt = [vec(dPhidt); vec(dBmdt); vec(dBpdt); drdt]

        return dVdt
    end

    # Continuous to discrete time dynamics conversion function
    discretize = (t2) -> begin
        # Propagate the dynamics over a single time interval
        Δt = τ2[2] - τ2[1]
        F = (t, V) -> derivs(t, V, Δt, tdil(t2))
        t_grid = RealVector(LinRange(0, Δt, 100))
        V0 = zeros(idcs_r[end])
        V0[idcs_A] = vec(I(nx))
        V = rk4(F, V0, t_grid)

        # Get the raw RK4 results
        AV = V[idcs_A]
        BmV = V[idcs_Bm]
        BpV = V[idcs_Bp]
        rV = V[idcs_r]

        # Extract the discrete-time update matrices for this time interval
        A = reshape(AV, (nx, nx))
        Bm = A * reshape(BmV, (nx, nu))
        Bp = A * reshape(BpV, (nx, nu))
        r = A * rV

        return A, Bm, Bp, r
    end

    # Variable scaling
    zero_intvl_tol = sqrt(eps())
    Tmax_x = veh.T_max1 * sin(traj.θmax2)

    update_scale! = (S, c, i, min, max) -> begin
        if min > max
            min, max = max, min
        end
        if (max - min) > zero_intvl_tol
            S[i, i] = max - min
            c[i] = min
        end
    end

    Sx, cx = RealMatrix(I(nx)), zeros(nx)
    Su, cu = RealMatrix(I(nu)), zeros(nu)

    update_scale!(Sx, cx, 1, 0, xs[veh.id_r[1]])
    update_scale!(Sx, cx, 2, 0, xs[veh.id_r[2]])
    update_scale!(Sx, cx, 3, 0, xs[veh.id_v[1]])
    update_scale!(Sx, cx, 4, 0, xs[veh.id_v[2]])
    update_scale!(Su, cu, 1, -Tmax_x, Tmax_x)
    update_scale!(Su, cu, 2, veh.T_min1, veh.T_max1)

    # Solver for a trajectory, given a time of flight
    solve_trajectory =
        (t2) -> begin
            # >> Formulate the convex optimization problem <<
            cvx = ConicProgram(
                nothing;
                solver = ECOS.Optimizer,
                solver_options = Dict("verbose" => 0),
            )

            # Decision variables
            x = @new_variable(cvx, (nx, N2), "x")
            u = @new_variable(cvx, (nu, N2), "u")
            @scale(x, diag(Sx), cx)
            @scale(u, diag(Su), cu)

            # Boundary conditions
            x0 = zeros(nx)
            xf = zeros(nx)
            x0[1:2] = xs[veh.id_r]
            x0[3:4] = xs[veh.id_v]
            xf[3:4] = traj.vf
            @add_constraint(
                cvx,
                ZERO,
                "initial_condition",
                (x[:, 1],),
                begin
                    local x0_var = arg[1]
                    x0_var - x0
                end
            )
            @add_constraint(
                cvx,
                ZERO,
                "final_condition",
                (x[:, end],),
                begin
                    local xf_var = arg[1]
                    xf_var - xf
                end
            )

            # Dynamics
            A, Bm, Bp, r = discretize(t2)
            for k = 1:N2-1
                xk, xkp1, uk, ukp1 = x[:, k], x[:, k+1], u[:, k], u[:, k+1]
                @add_constraint(
                    cvx,
                    ZERO,
                    "dynamics",
                    (xkp1, xk, uk, ukp1),
                    begin
                        local xn, x, u, un = arg
                        xn - (A * x + Bm * u + Bp * un + r)
                    end
                )
            end

            # Input constraints
            for k = 1:N2
                uk = u[:, k]
                @add_constraint(cvx, SOC, "max_thrust", (uk,), begin
                    local uk = arg[1]
                    vcat(veh.T_max1, uk)
                end)
                @add_constraint(
                    cvx,
                    NONPOS,
                    "pointy_end_up",
                    (uk,),
                    begin
                        local uk = arg[1]
                        veh.T_min1 - dot(uk, env.ey)
                    end
                )
                @add_constraint(
                    cvx,
                    SOC,
                    "tilt",
                    (uk,),
                    begin
                        local uk = arg[1]
                        vcat(dot(uk, env.ey) / cos(traj.θmax2), uk)
                    end
                )
            end

            # State constraints
            for k = 1:N2
                xk = x[:, k]
                rk = xk[1:2]
                # acc!(cvx, C(vcat(dot(rk, env.ey)/cos(traj.γ_gs), rk), :soc))
                @add_constraint(cvx, NONPOS, "above_ground", (rk,), begin
                        local rk = arg[1]
                        -dot(rk, env.ey)
                    end)
            end

            # >> Solve <<
            status = solve!(cvx)

            # Return the solution
            x = value(x)
            u = value(u)

            return x, u, status
        end

    # Find the first (smallest) time that gives a feasible trajectory
    t2_range = [10.0, 40.0]
    Δt2 = 1.0 # Amount to increment t2 guess by
    t2, x2, T2 = t2_range[1], nothing, nothing
    while true
        @printf(".")
        _x, _u, status = solve_trajectory(t2)
        if status == MOI.OPTIMAL || status == MOI.ALMOST_OPTIMAL
            x2 = _x
            T2 = _u
            break
        end
        t2 += Δt2
        if t2 > t2_range[2]
            msg = string("could not find a terminal ", "descent time of flight.")
            err = SCPError(0, SCP_BAD_PROBLEM, msg)
            throw(err)
        end
    end

    # Add terminal descent to initial guess
    x_guess[veh.id_r, id_phase2] = x2[1:2, :]
    x_guess[veh.id_v, id_phase2] = x2[3:4, :]
    _tdil = tdil(t2)
    m20 = x_guess[veh.id_m, id_phase2[1]]
    for k = 1:N2
        Tk = T2[:, k]
        j = id_phase2[k]
        x_guess[veh.id_θ, j] = -atan(Tk[1], Tk[2])
        u_guess[veh.id_T, j] = norm(Tk)
        if k > 1
            # Angular velocity
            Δθ = x_guess[veh.id_θ, j] - x_guess[veh.id_θ, j-1]
            Δt = (τ2[k] - τ2[k-1]) * _tdil
            x_guess[veh.id_ω, j-1] = Δθ / Δt
            # Mass
            x_guess[veh.id_m, j] =
                m20 + trapz(veh.αe * u_guess[veh.id_T, id_phase2[1:k]], τ2[1:k] * _tdil)
        end
    end

    # Parameter guess
    p_guess = RealVector(undef, pbm.np)
    p_guess[veh.id_t1] = t1
    p_guess[veh.id_t2] = t2
    p_guess[veh.id_xs] = xs

    @printf(". done\n")

    return x_guess, u_guess, p_guess
end

function set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(pbm, starship_initial_guess)

    return nothing
end

function set_cost!(pbm::TrajectoryProblem)::Nothing

    problem_set_terminal_cost!(pbm, (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        env = pbm.mdl.env
        # Altitude at phase switch
        # Goal: maximize it
        rs = p[veh.id_xs][veh.id_r]
        alt = dot(rs, env.ey)
        alt_nrml = traj.hs
        alt_cost = -alt / alt_nrml
        μ = 0.3 # Relative weight to fuel cost
        # Fuel consumption
        # Goal: minimize it
        mf = x[veh.id_m]
        Δm = 0.0 - mf
        Δm_nrml = 10e3
        Δm_cost = Δm / Δm_nrml
        # Total cost
        return μ * alt_cost + Δm_cost
    end)

    return nothing
end

"""
    dynamics(t, k, x, u, p, pbm[; no_aero_torques])

Starship vehicle dynamics.

Args:
- `t`: the current time (normalized).
- `k`: the current discrete-time node.
- `x`: the current state vector.
- `u`: the current input vector.
- `p`: the parameter vector.
- `pbm`: the Starship landing flip problem description.
- `no_aero_torques`: (optional) whether to omit torques generated by lift and
  drag.

Returns:
- `f`: the time derivative of the state vector.
"""
function dynamics(
    t::RealValue,
    k::Int,
    x::RealVector,
    u::RealVector,
    p::RealVector,
    pbm::TrajectoryProblem;
    no_aero_torques::Bool = false,
)::RealVector

    # Parameters
    veh = pbm.mdl.vehicle
    env = pbm.mdl.env
    traj = pbm.mdl.traj

    # Current (x, u, p) values
    v = x[veh.id_v]
    θ = x[veh.id_θ]
    ω = x[veh.id_ω]
    m = x[veh.id_m]
    δd = x[veh.id_δd]
    T = u[veh.id_T]
    δ = u[veh.id_δ]
    tdil = ((t <= traj.τs) ? p[veh.id_t1] / traj.τs : p[veh.id_t2] / (1 - traj.τs))

    # Derived quantities
    ℓeng = -veh.lcg
    ℓcp = veh.lcp - veh.lcg
    ei = veh.ei(θ)
    ej = veh.ej(θ)
    Tv = T * (-sin(δ) * ei + cos(δ) * ej)
    MT = ℓeng * T * sin(δ)
    D = -veh.CD * norm(v) * v
    if !no_aero_torques
        MD = -ℓcp * dot(D, ei)
    else
        MD = 0.0
    end

    # The dynamics
    f = zeros(pbm.nx)
    f[veh.id_r] = v
    f[veh.id_v] = (Tv + D) / veh.m + env.g
    f[veh.id_θ] = ω
    f[veh.id_ω] = (MT + MD) / veh.J
    f[veh.id_m] = veh.αe * T
    f[veh.id_δd] = (δ - δd) / veh.rate_delay

    # Scale for time
    f *= tdil

    return f
end

function set_dynamics!(pbm::TrajectoryProblem)::Nothing

    problem_set_dynamics!(
        pbm,
        # Dynamics f
        (t, k, x, u, p, pbm) -> begin
            f = dynamics(t, k, x, u, p, pbm)
            return f
        end,
        # Jacobian df/dx
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj
            v = x[veh.id_v]
            θ = x[veh.id_θ]
            m = x[veh.id_m]
            T = u[veh.id_T]
            δ = u[veh.id_δ]
            tdil = ((t <= traj.τs) ? p[veh.id_t1] / traj.τs : p[veh.id_t2] / (1 - traj.τs))

            ℓcp = veh.lcp - veh.lcg
            ei = veh.ei(θ)
            ej = veh.ej(θ)
            D = -veh.CD * norm(v) * v
            ∇θ_Tv = T * (-sin(δ) * ej + cos(δ) * -ei)
            ∇v_D = -veh.CD * (norm(v) * I(2) + (v * v') / norm(v))
            ∇v_MD = -ℓcp * ∇v_D' * ei
            ∇θ_MD = -ℓcp * dot(D, ej)

            A = zeros(pbm.nx, pbm.nx)
            A[veh.id_r, veh.id_v] = I(2)
            A[veh.id_v, veh.id_v] = (∇v_D) / veh.m
            A[veh.id_v, veh.id_θ] = (∇θ_Tv) / veh.m
            A[veh.id_θ, veh.id_ω] = 1.0
            A[veh.id_ω, veh.id_v] = ∇v_MD / veh.J
            A[veh.id_ω, veh.id_θ] = ∇θ_MD / veh.J
            A[veh.id_δd, veh.id_δd] = -1.0 / veh.rate_delay

            A *= tdil
            return A
        end,
        # Jacobian df/du
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj
            v = x[veh.id_v]
            θ = x[veh.id_θ]
            m = x[veh.id_m]
            T = u[veh.id_T]
            δ = u[veh.id_δ]
            tdil = ((t <= traj.τs) ? p[veh.id_t1] / traj.τs : p[veh.id_t2] / (1 - traj.τs))

            ℓeng = -veh.lcg
            ei = veh.ei(θ)
            ej = veh.ej(θ)
            ∇T_Tv = -sin(δ) * ei + cos(δ) * ej
            ∇δ_Tv = T * (-cos(δ) * ei - sin(δ) * ej)
            ∇T_MT = ℓeng * sin(δ)
            ∇δ_MT = ℓeng * T * cos(δ)

            B = zeros(pbm.nx, pbm.nu)
            B[veh.id_v, veh.id_T] = (∇T_Tv) / veh.m
            B[veh.id_v, veh.id_δ] = (∇δ_Tv) / veh.m
            B[veh.id_ω, veh.id_T] = (∇T_MT) / veh.J
            B[veh.id_ω, veh.id_δ] = (∇δ_MT) / veh.J
            B[veh.id_m, veh.id_T] = veh.αe
            B[veh.id_δd, veh.id_δ] = 1.0 / veh.rate_delay

            B *= tdil
            return B
        end,
        # Jacobian df/dp
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            id_t = (t <= traj.τs) ? veh.id_t1 : veh.id_t2
            F = zeros(pbm.nx, pbm.np)
            F[:, id_t] = pbm.f(t, k, x, u, p) / p[id_t]
            return F
        end,
    )

    return nothing
end

function set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the state
    problem_set_X!(
        pbm,
        (t, k, x, p, pbm, ocp) -> begin
            traj = pbm.mdl.traj
            env = pbm.mdl.env
            veh = pbm.mdl.vehicle
            r = x[veh.id_r]
            v = x[veh.id_v]
            tf1 = p[veh.id_t1]
            tf2 = p[veh.id_t2]

            @add_constraint(ocp, NONPOS, "no_climb", (v,), begin
                local v = arg[1]
                dot(v, env.ey)
            end)

            @add_constraint(
                ocp,
                NONPOS,
                "max_time",
                (tf1, tf2),
                begin
                    local tf1, tf2 = arg
                    local tf = tf1[1] + tf2[1]
                    tf - traj.tf_max
                end
            )

            @add_constraint(
                ocp,
                NONPOS,
                "min_time",
                (tf1, tf2),
                begin
                    local tf1, tf2 = arg
                    local tf = tf1[1] + tf2[1]
                    traj.tf_min - tf
                end
            )
        end,
    )

    # Convex path constraints on the input
    problem_set_U!(
        pbm,
        (t, k, u, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            T = u[veh.id_T]
            δ = u[veh.id_δ]
            flip_phase = t <= traj.τs
            T_max = (flip_phase) ? veh.T_max3 : veh.T_max1
            T_min = (flip_phase) ? veh.T_min3 : veh.T_min1

            @add_constraint(ocp, NONPOS, "max_thrust", (T,), begin
                local T = arg[1]
                T[1] - T_max
            end)

            @add_constraint(ocp, NONPOS, "min_thrust", (T,), begin
                local T = arg[1]
                T_min - T[1]
            end)

            @add_constraint(ocp, L1, "gimbal", (δ,), begin
                local δ = arg[1]
                vcat(veh.δ_max, δ)
            end)
        end,
    )

    return nothing
end

function set_nonconvex_constraints!(pbm::TrajectoryProblem, algo::Symbol)::Nothing

    # Return true if this is the temporal node where phase 1 ends
    phase_switch =
        (t, pbm) -> begin
            Δt = 1 / (pbm.scp.N - 1)
            τs = pbm.mdl.traj.τs
            tol = 1e-3
            local is_phase_switch = (τs - Δt) + tol <= t && t <= τs + tol
            return is_phase_switch
        end

    # Return true if this is a phase 2 temporal node
    phase2 = (t, pbm) -> begin
        τs = pbm.mdl.traj.τs
        local is_phase2 = phase_switch(t, pbm) || t > τs
        return is_phase2
    end

    _common_s_sz = 7 + 2 * pbm.nx

    problem_set_s!(
        pbm,
        algo,
        # Constraint s
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj
            r = x[veh.id_r]
            θ = x[veh.id_θ]
            δd = x[veh.id_δd]
            δ = u[veh.id_δ]
            δdot = u[veh.id_δdot]

            s = zeros(_common_s_sz)
            s[1] = (δ - δd) - δdot * veh.rate_delay
            s[2] = δdot * veh.rate_delay - (δ - δd)
            s[3] = δdot - veh.δdot_max
            s[4] = -veh.δdot_max - δdot
            s[5] = norm(r) * cos(traj.γ_gs) - dot(r, env.ey)
            if phase_switch(t, pbm)
                s[(1:pbm.nx).+5] = p[veh.id_xs] - x
                s[(1:pbm.nx).+(5+pbm.nx)] = x - p[veh.id_xs]
            end
            if phase2(t, pbm)
                s[end-1] = θ - traj.θmax2
                s[end] = -traj.θmax2 - θ
            end

            return s
        end,
        # Jacobian ds/dx
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj
            r = x[veh.id_r]

            nrm_r = norm(r)
            ∇nrm_r = (nrm_r < sqrt(eps())) ? zeros(2) : r / nrm_r

            C = zeros(_common_s_sz, pbm.nx)
            C[1, veh.id_δd] = -1.0
            C[2, veh.id_δd] = 1.0
            C[5, veh.id_r] = ∇nrm_r * cos(traj.γ_gs) - env.ey
            if phase_switch(t, pbm)
                C[(1:pbm.nx).+5, :] = -I(pbm.nx)
                C[(1:pbm.nx).+(5+pbm.nx), :] = I(pbm.nx)
            end
            if phase2(t, pbm)
                C[end-1, veh.id_θ] = 1.0
                C[end, veh.id_θ] = -1.0
            end

            return C
        end,
        # Jacobian ds/du
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle

            D = zeros(_common_s_sz, pbm.nu)
            D[1, veh.id_δ] = 1.0
            D[1, veh.id_δdot] = -veh.rate_delay
            D[2, veh.id_δ] = -1.0
            D[2, veh.id_δdot] = veh.rate_delay
            D[3, veh.id_δdot] = 1.0
            D[4, veh.id_δdot] = -1.0

            return D
        end,
        # Jacobian ds/dp
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle

            G = zeros(_common_s_sz, pbm.np)
            if phase_switch(t, pbm)
                G[(1:pbm.nx).+5, veh.id_xs] = I(pbm.nx)
                G[(1:pbm.nx).+(5+pbm.nx), veh.id_xs] = -I(pbm.nx)
            end

            return G
        end,
    )

    return nothing
end

function set_bcs!(pbm::TrajectoryProblem)::Nothing

    # Initial conditions
    problem_set_bc!(
        pbm,
        :ic,
        # Constraint g
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            rhs = zeros(7)
            rhs[1:2] = traj.r0
            rhs[3:4] = traj.v0
            rhs[5] = traj.θ0
            rhs[6] = 0.0
            rhs[7] = 0.0
            g = x[vcat(veh.id_r, veh.id_v, veh.id_θ, veh.id_ω, veh.id_m)] - rhs
            return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            H = zeros(7, pbm.nx)
            H[1:2, veh.id_r] = I(2)
            H[3:4, veh.id_v] = I(2)
            H[5, veh.id_θ] = 1.0
            H[6, veh.id_ω] = 1.0
            H[7, veh.id_m] = 1.0
            return H
        end,
    )

    # Terminal conditions
    problem_set_bc!(
        pbm,
        :tc,
        # Constraint g
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            rhs = zeros(6)
            rhs[1:2] = zeros(2)
            rhs[3:4] = traj.vf
            rhs[5] = 0.0
            rhs[6] = 0.0
            g = x[vcat(veh.id_r, veh.id_v, veh.id_θ, veh.id_ω)] - rhs
            return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            H = zeros(6, pbm.nx)
            H[1:2, veh.id_r] = I(2)
            H[3:4, veh.id_v] = I(2)
            H[5, veh.id_θ] = 1.0
            H[6, veh.id_ω] = 1.0
            return H
        end,
    )

    return nothing
end
