#= 6-Degree of Freedom free-flyer example, common code.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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

include("../../models/freeflyer.jl")
include("../../core/problem.jl")
include("../../utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function define_problem!(pbm::TrajectoryProblem)::Nothing
    _common__set_dims!(pbm)
    _common__set_scale!(pbm)
    _common__set_terminal_cost!(pbm)
    _common__set_convex_constraints!(pbm)
    _common__set_bcs!(pbm)

    _common__set_guess!(pbm)

    return nothing
end

function _common__set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 13, 6, 1)

    return nothing
end

function _common__set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl

    veh, traj = mdl.vehicle, mdl.traj
    for i in veh.id_r
        min_pos = min(traj.r0[i], traj.rf[i])
        max_pos = max(traj.r0[i], traj.rf[i])
        problem_advise_scale!(pbm, :state, i, (min_pos, max_pos))
    end
    problem_advise_scale!(pbm, :parameter, veh.id_t, (traj.tf_min, traj.tf_max))

    return nothing
end

function _common__set_guess!(pbm::TrajectoryProblem)::Nothing

    # Use an L-shaped axis-aligned position trajectory, a corresponding
    # velocity trajectory, a SLERP interpolation for the quaternion attitude
    # and a corresponding constant-speed angular velocity.

    problem_set_guess!(
        pbm,
        (N, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        # No existing reference provided - make a new guess
        # >> Parameter guess <<
        p = zeros(pbm.np)
        flight_time = 0.5*(traj.tf_min+traj.tf_max)
        p[veh.id_t] = flight_time
        # >> State guess <<
        x = T_RealMatrix(undef, pbm.nx, N)
        # @ Position/velocity L-shape trajectory @
        Δτ = flight_time/(N-1)
        speed = norm(traj.rf-traj.r0, 1)/flight_time
        times = straightline_interpolate([0.0], [flight_time],
                                         N)
        flight_time_leg = abs.(traj.rf-traj.r0)/speed
        flight_time_leg_cumul = cumsum(flight_time_leg)
        r = view(x, veh.id_r, :)
        v = view(x, veh.id_v, :)
        for k = 1:N
        # --- for k
        tk = @k(times)[1]
        for i = 1:3
        # -- for i
        if tk <= flight_time_leg_cumul[i]
        # - if tk
        # Current node is in i-th leg of the trajectory
        # Endpoint times
        t0 = (i>1) ? flight_time_leg_cumul[i-1] : 0.0
        tf = flight_time_leg_cumul[i]
        # Endpoint positions
        r0 = copy(traj.r0)
        r0[1:i-1] = traj.rf[1:i-1]
        rf = copy(r0)
        rf[i] = traj.rf[i]
        @k(r) = linterp(tk, hcat(r0, rf), [t0, tf])
        # Velocity
        dir_vec = rf-r0
        dir_vec /= norm(dir_vec)
        v_leg = speed*dir_vec
        @k(v) = v_leg
        break
        # - if tk
        end
        # -- for i
        end
        # --- for k
        end
        # @ Quaternion SLERP interpolation @
        x[veh.id_q, :] = T_RealMatrix(undef, 4, N)
        for k = 1:N
        mix = (k-1)/(N-1)
        @k(view(x, veh.id_q, :)) = vec(slerp_interpolate(
            traj.q0, traj.qf, mix))
        end
        # @ Constant angular velocity @
        rot_ang, rot_ax = Log(traj.qf*traj.q0')
        rot_speed = rot_ang/flight_time
        ang_vel = rot_speed*rot_ax
        x[veh.id_ω, :] = straightline_interpolate(
            ang_vel, ang_vel, N)
        # >> Input guess <<
        idle = zeros(pbm.nu)
        u = straightline_interpolate(idle, idle, N)
        return x, u, p
        end)

    return nothing
end

function _common__set_terminal_cost!(pbm::TrajectoryProblem)::Nothing

    problem_set_terminal_cost!(
        pbm, (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        tdil = p[veh.id_t]
        tdil_max = traj.tf_max
        γ = traj.γ
        return γ*(tdil/tdil_max)^2
        end)

    return nothing
end

function _common__set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the state
    problem_set_X!(
        pbm, (τ, x, pbm) -> begin
        traj = pbm.mdl.traj
        veh = pbm.mdl.vehicle
        C = T_ConvexConeConstraint
        X = [C(vcat(veh.v_max, x[veh.id_v]), :soc),
             C(vcat(veh.ω_max, x[veh.id_ω]), :soc)]
        return X
        end)

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (τ, u, pbm) -> begin
        veh = pbm.mdl.vehicle
        C = T_ConvexConeConstraint
        U = [C(vcat(veh.T_max, u[veh.id_T]), :soc),
             C(vcat(veh.M_max, u[veh.id_M]), :soc)]
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
        rhs = zeros(pbm.nx)
        rhs[veh.id_r] = traj.r0
        rhs[veh.id_v] = traj.v0
        rhs[veh.id_q] = vec(traj.q0)
        rhs[veh.id_ω] = traj.ω0
        g = x-rhs
        return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
        H = I(pbm.nx)
        return H
        end,
        # Jacobian dg/dp
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        K = zeros(pbm.nx, pbm.np)
        return K
        end)

    # Terminal conditions
    problem_set_bc!(
        pbm, :tc,
        # Constraint g
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        rhs = zeros(pbm.nx)
        rhs[veh.id_r] = traj.rf
        rhs[veh.id_v] = traj.vf
        rhs[veh.id_q] = vec(traj.qf)
        rhs[veh.id_ω] = traj.ωf
        g = x-rhs
        return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
        H = I(pbm.nx)
        return H
        end,
        # Jacobian dg/dp
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        K = zeros(pbm.nx, pbm.np)
        return K
        end)

    return nothing
end
