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

include("../../models/oscillator.jl")
include("../../core/problem.jl")
include("../../utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function define_problem!(pbm::TrajectoryProblem,
                         algo::T_Symbol)::Nothing
    _common__set_dims!(pbm)
    _common__set_scale!(pbm)
    _common__set_cost!(pbm, algo)
    _common__set_dynamics!(pbm)
    _common__set_convex_constraints!(pbm)
    # _common__set_nonconvex_constraints!(pbm, algo)
    _common__set_bcs!(pbm)

    _common__set_guess!(pbm)

    return nothing
end

function _common__set_dims!(pbm::TrajectoryProblem)::Nothing

    # Parameters
    np = pbm.mdl.vehicle.id_l1r[end]

    problem_set_dims!(pbm, 2, 1, np)

    return nothing
end

function _common__set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl
    veh = mdl.vehicle
    traj = mdl.traj

    advise! = problem_advise_scale!

    # States
    advise!(pbm, :state, veh.id_r, (-traj.r0, traj.r0))
    advise!(pbm, :state, veh.id_v, (-traj.v0, traj.v0))
    # Inputs
    advise!(pbm, :input, veh.id_a, (-veh.a_max, veh.a_max))
    # Parameters
    for i in veh.id_l1r
        advise!(pbm, :parameter, i, (0.0, traj.r0))
    end

    return nothing
end

function _common__set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(
        pbm, (N, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj

        # >> State guess <<
        x0 = zeros(pbm.nx)
        x0[veh.id_r] = traj.r0
        x0[veh.id_v] = traj.v0

        t_grid = T_RealVector(LinRange(0.0, 1.0, 1000))
        τ_grid = T_RealVector(LinRange(0.0, 1.0, N))

        F = (t, x) -> begin
        k = max(floor(T_Int, t/(N-1))+1, N)
        u = zeros(pbm.nu)
        p = zeros(pbm.np)
        dxdt = dynamics(t, k, x, u, p, pbm)
        return dxdt
        end

        X = rk4(F, x0, t_grid; full=true)
        Xc = T_ContinuousTimeTrajectory(t_grid, X, :linear)
        x = hcat([sample(Xc, τ) for τ in τ_grid]...)

        # >> Input guess <<
        idle = zeros(pbm.nu)
        u = straightline_interpolate(idle, idle, N)

        # >> Parameter guess <<
        p = zeros(pbm.np)
        for k = 1:N
        @k(p) = norm(@k(x)[veh.id_r], 1)
        end

        return x, u, p
        end)

    return nothing
end

function _common__set_cost!(pbm::TrajectoryProblem,
                            algo::T_Symbol)::Nothing

    problem_set_running_cost!(
        pbm, algo,
        (t, k, x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        l1r = @k(p[veh.id_l1r])
        a = u[veh.id_a]
        r_nrml = traj.r0
        a_nrml = veh.a_max
        α = 0.06 # Tradeoff
        return α*(a/a_nrml)^2+(l1r/r_nrml)
        end)

    return nothing
end

function _common__set_dynamics!(pbm::TrajectoryProblem)::Nothing

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
        A[veh.id_v, veh.id_v] = -2*veh.ζ*veh.ω0
        A *= traj.tf
        return A
        end,
        # Jacobian df/du
        (t, k, x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        B = zeros(pbm.nx, pbm.nu)
        B[veh.id_v, veh.id_a] = 1.0
        B *= traj.tf
        return B
        end,
        # Jacobian df/dp
        (t, k, x, u, p, pbm) -> begin
        F = zeros(pbm.nx, pbm.np)
        return F
        end,)

    return nothing
end

function _common__set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    problem_set_X!(
        pbm, (t, k, x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        N = pbm.scp.N

        r = x[veh.id_r]
        l1r = p[veh.id_l1r]

        C = T_ConvexConeConstraint
        X = [C(vcat(@k(l1r), r), :l1)]

        return X
        end)

    problem_set_U!(
        pbm, (t, k, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle

        a = u[veh.id_a]

        C = T_ConvexConeConstraint
        U = [C(a-veh.a_max, :nonpos),
             C(-veh.a_max-a, :nonpos)]

        return U
        end)

    return nothing
end

function _common__set_bcs!(pbm::TrajectoryProblem)::Nothing

    # Initial conditions
    problem_set_bc!(
        pbm, :ic,
        # Constraint g
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        rhs = zeros(2)
        rhs[1] = traj.r0
        rhs[2] = traj.v0
        g = x[vcat(veh.id_r,
                   veh.id_v)]-rhs
        return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        H = zeros(2, pbm.nx)
        H[1, veh.id_r] = 1.0
        H[2, veh.id_v] = 1.0
        return H
        end)

    return nothing
end
