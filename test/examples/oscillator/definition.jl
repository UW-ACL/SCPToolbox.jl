#= Forced harmonic oscillator with input deadband, common code.

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

# ..:: Methods ::..

function define_problem!(pbm::TrajectoryProblem, algo::Symbol)::Nothing

    set_dims!(pbm)
    set_scale!(pbm)
    set_cost!(pbm, algo)
    set_dynamics!(pbm)
    set_convex_constraints!(pbm)
    set_nonconvex_constraints!(pbm, algo)
    set_bcs!(pbm)

    set_guess!(pbm)

    return nothing
end

function set_dims!(pbm::TrajectoryProblem)::Nothing

    # Parameters
    np = pbm.mdl.vehicle.id_l1r[end]

    problem_set_dims!(pbm, 2, 4, np)

    return nothing
end

function set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl
    veh = mdl.vehicle
    traj = mdl.traj

    advise! = problem_advise_scale!

    # States
    advise!(pbm, :state, veh.id_r, (-traj.r0, traj.r0))
    advise!(pbm, :state, veh.id_v, (-traj.v0, traj.v0))

    # Inputs
    advise!(pbm, :input, veh.id_aa, (-veh.a_max, veh.a_max))
    advise!(pbm, :input, veh.id_ar, (-veh.a_max, veh.a_max))
    advise!(pbm, :input, veh.id_l1aa, (0.0, veh.a_max))
    advise!(pbm, :input, veh.id_l1adiff, (0.0, 2 * veh.a_max))

    # Parameters
    advise!(pbm, :parameter, veh.id_l1r, (0.0, traj.r0))

    return nothing
end

function set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(
        pbm,
        (N, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            # >> State guess <<
            x0 = zeros(pbm.nx)
            x0[veh.id_r] = traj.r0
            x0[veh.id_v] = traj.v0

            t_grid = RealVector(LinRange(0.0, 1.0, 1000))
            τ_grid = RealVector(LinRange(0.0, 1.0, N))

            F = (t, x) -> begin
                k = max(floor(Int, t / (N - 1)) + 1, N)
                u = zeros(pbm.nu)
                p = zeros(pbm.np)
                dxdt = dynamics(t, k, x, u, p, pbm)
                return dxdt
            end

            X = rk4(F, x0, t_grid; full = true)
            Xc = ContinuousTimeTrajectory(t_grid, X, :linear)
            x = hcat([sample(Xc, τ) for τ in τ_grid]...)

            # >> Input guess <<
            idle = zeros(pbm.nu)
            u = straightline_interpolate(idle, idle, N)

            # >> Parameter guess <<
            p = zeros(pbm.np)
            for k = 1:N
                p[k] = norm(x[veh.id_r, k], 1)
            end

            return x, u, p
        end,
    )

    return nothing
end

function set_cost!(pbm::TrajectoryProblem, algo::Symbol)::Nothing

    problem_set_running_cost!(
        pbm,
        algo,
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            l1aa = u[veh.id_l1aa]
            l1adiff = u[veh.id_l1adiff]
            l1r = p[veh.id_l1r[k]]

            r_nrml = traj.r0
            a_nrml = veh.a_max

            # Compute value
            f = l1r / r_nrml
            f += traj.α * l1aa / a_nrml
            f += traj.γ * l1adiff / a_nrml

            return f
        end,
    )

    return nothing
end


"""
    dynamics(t, k, x, u, p, pbm)

Oscillator dynamics.

# Arguments
- `t`: the current time (normalized).
- `k`: the current discrete-time node.
- `x`: the current state vector.
- `u`: the current input vector.
- `p`: the parameter vector.
- `pbm`: the oscillator problem description.

# Returns
- `f`: the time derivative of the state vector.
"""
function dynamics(
    t::RealValue,
    k::Int,
    x::RealVector,
    u::RealVector,
    p::RealVector,
    pbm::TrajectoryProblem,
)::RealVector

    impulse = k < 0

    # Parameters
    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj

    # Current (x, u, p) values
    r = x[veh.id_r]
    v = x[veh.id_v]
    aa = u[veh.id_aa]

    # The dynamics
    f = zeros(pbm.nx)
    f[veh.id_v] = aa
    if !impulse
        f[veh.id_r] = v
        f[veh.id_v] += -veh.ω0^2 * r - 2 * veh.ζ * veh.ω0 * v
        # Scale for absolute time
        f *= traj.tf
    end

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
            traj = pbm.mdl.traj
            A = zeros(pbm.nx, pbm.nx)
            A[veh.id_r, veh.id_v] = 1.0
            A[veh.id_v, veh.id_r] = -veh.ω0^2
            A[veh.id_v, veh.id_v] = -2 * veh.ζ * veh.ω0
            # Scale for absolute time
            A *= traj.tf
            return A
        end,
        # Jacobian df/du
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            impulse = k < 0
            B = zeros(pbm.nx, pbm.nu)
            B[veh.id_v, veh.id_aa] = 1.0
            if !impulse
                # Scale for absolute time
                B *= traj.tf
            end
            return B
        end,
        # Jacobian df/dp
        (t, k, x, u, p, pbm) -> begin
            F = zeros(pbm.nx, pbm.np)
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
            veh = pbm.mdl.vehicle
            N = pbm.scp.N

            r = x[veh.id_r]
            l1r_k = (p[veh.id_l1r])[k]

            @add_constraint(
                ocp,
                L1,
                "abs_r",
                (r, l1r_k),
                begin # Value
                    local r, l1r_k = arg
                    vcat(l1r_k, r)
                end,
                begin # Jacobians
                    J[1] = vcat(zeros(length(r))', I(length(r)))
                    J[2] = vcat(1, zeros(length(r)))
                end
            )
        end,
    )

    # Convex path constraints on the input
    problem_set_U!(
        pbm,
        (t, k, u, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle

            aa = u[veh.id_aa]
            ar = u[veh.id_ar]
            l1aa = u[veh.id_l1aa]
            l1adiff = u[veh.id_l1adiff]

            @add_constraint(
                ocp,
                NONPOS,
                "accel_bounds",
                (aa,),
                begin # Value
                    local aa, = arg
                    aa - veh.a_max
                end,
                begin # Jacobians
                    J[1] = [1]
                end
            )

            @add_constraint(
                ocp,
                NONPOS,
                "accel_bounds",
                (aa,),
                begin # Value
                    local aa, = arg
                    -veh.a_max - aa
                end,
                begin # Jacobians
                    J[1] = [-1]
                end
            )

            @add_constraint(
                ocp,
                NONPOS,
                "accel_bounds",
                (ar,),
                begin # Value
                    local ar, = arg
                    ar - veh.a_max
                end,
                begin # Jacobians
                    J[1] = [1]
                end
            )

            @add_constraint(
                ocp,
                NONPOS,
                "accel_bounds",
                (ar,),
                begin # Value
                    local ar, = arg
                    -veh.a_max - ar
                end,
                begin # Jacobians
                    J[1] = [-1]
                end
            )

            @add_constraint(
                ocp,
                L1,
                "accel_bounds",
                (l1aa, aa),
                begin # Value
                    local l1aa, aa = arg
                    vcat(l1aa, aa)
                end,
                begin # Jacobians
                    J[1] = [1; 0]
                    J[2] = [0; 1]
                end
            )

            @add_constraint(
                ocp,
                L1,
                "accel_bounds",
                (l1adiff, aa, ar),
                begin # Value
                    local l1adiff, aa, ar = arg
                    vcat(l1adiff, aa - ar)
                end,
                begin # Jacobians
                    J[1] = [1; 0]
                    J[2] = [0; 1]
                    J[3] = [0; -1]
                end
            )
        end,
    )

    return nothing
end

function set_nonconvex_constraints!(pbm::TrajectoryProblem, algo::Symbol)::Nothing

    _common_s_sz = 2

    problem_set_s!(
        pbm,
        algo,
        # Constraint s
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            aa = u[veh.id_aa]
            ar = u[veh.id_ar]

            above_db = ar - veh.a_db
            below_db = -veh.a_db - ar
            OR = or(
                [above_db; below_db],
                κ = traj.κ1,
                match = veh.a_max - veh.a_db,
                normalize = veh.a_max - veh.a_db,
            )

            s = zeros(_common_s_sz)

            s[1] = aa - OR * ar
            s[2] = OR * ar - aa

            return s
        end,
        # Jacobian ds/dx
        (t, k, x, u, p, pbm) -> begin
            C = zeros(_common_s_sz, pbm.nx)
            return C
        end,
        # Jacobian ds/du
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            aa = u[veh.id_aa]
            ar = u[veh.id_ar]

            above_db = ar - veh.a_db
            ∇above_db = [1.0]
            below_db = -veh.a_db - ar
            ∇below_db = [-1.0]
            OR, ∇OR = or(
                [above_db; below_db],
                [∇above_db, ∇below_db],
                κ = traj.κ1,
                match = veh.a_max - veh.a_db,
                normalize = veh.a_max - veh.a_db,
            )
            ∇ORar = ∇OR[1] * ar + OR

            D = zeros(_common_s_sz, pbm.nu)

            D[1, veh.id_aa] = 1.0
            D[1, veh.id_ar] = -∇ORar
            D[2, veh.id_aa] = -1.0
            D[2, veh.id_ar] = ∇ORar

            return D
        end,
        # Jacobian ds/dp
        (t, k, x, u, p, pbm) -> begin
            G = zeros(_common_s_sz, pbm.np)
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
            rhs = zeros(2)
            rhs[1] = traj.r0
            rhs[2] = traj.v0
            g = x[vcat(veh.id_r, veh.id_v)] - rhs
            return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            H = zeros(2, pbm.nx)
            H[1, veh.id_r] = 1.0
            H[2, veh.id_v] = 1.0
            return H
        end,
    )

    return nothing
end
