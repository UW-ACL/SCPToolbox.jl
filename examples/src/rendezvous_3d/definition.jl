#= Spacecraft rendezvous problem definition.

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

LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("parameters.jl")

    include("../../../parser/src/Parser.jl")

    using .Parser
    import .Parser.ConicLinearProgram: @add_constraint
    import .Parser.ConicLinearProgram: ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
    import .Utils.Types: slerp_interpolate, Log, skew
end

using Parser

# ..:: Methods ::..

function define_problem!(pbm::TrajectoryProblem,
                         algo::Symbol)::Nothing

    set_dims!(pbm)
    set_scale!(pbm)
    set_integration!(pbm)
    set_cost!(pbm, algo)
    set_dynamics!(pbm)
    set_convex_constraints!(pbm)
    set_nonconvex_constraints!(pbm, algo)
    set_bcs!(pbm)

    set_guess!(pbm)

    return nothing
end # function

function set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 13, 22, 1)

    return nothing
end # function

function set_scale!(pbm::TrajectoryProblem)::Nothing

    # Parameters
    mdl = pbm.mdl
    veh = mdl.vehicle
    traj = mdl.traj

    advise! = problem_advise_scale!

    # >> States <<
    r0_nrm = norm(traj.r0)
    v_max = r0_nrm/traj.tf_min
    rot_ang, _ = Log(traj.q0'*traj.qf)
    ω_max = abs(rot_ang)/traj.tf_min
    advise!(pbm, :state, veh.id_r, (-r0_nrm, r0_nrm))
    advise!(pbm, :state, veh.id_v, (-v_max, v_max))
    advise!(pbm, :state, veh.id_ω, (-ω_max, ω_max))

    # >> Inputs <<
    advise!(pbm, :input, veh.id_T, (-veh.T_max, veh.T_max))
    advise!(pbm, :input, veh.id_M, (-veh.M_max, veh.M_max))
    advise!(pbm, :input, veh.id_rcs, (-veh.csm.imp_max, veh.csm.imp_max))

    # >> Parameters <<
    advise!(pbm, :parameter, veh.id_t, (traj.tf_min, traj.tf_max))

    return nothing
end # function

function set_integration!(pbm::TrajectoryProblem)::Nothing

    # Quaternion re-normalization on numerical integration step
    problem_set_integration_action!(
        pbm, pbm.mdl.vehicle.id_q,
        (q, pbm) -> begin
            qn = q/norm(q)
            return qn
        end)

    return nothing
end # function

function set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(
        pbm, (N, pbm) -> begin

            # Parameters
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj

            # >> Parameter guess <<
            p = zeros(pbm.np)
            flight_time = 0.5*(traj.tf_min+traj.tf_max)
            p[veh.id_t] = flight_time

            # >> Input guess <<
            coast = zeros(pbm.nu)
            u = straightline_interpolate(coast, coast, N)

            # >> State guess <<
            x = RealMatrix(undef, pbm.nx, N)
            x[veh.id_r, :] = straightline_interpolate(traj.r0, traj.rf, N)
            v_cst = (traj.rf-traj.r0)/flight_time
            x[veh.id_v, :] = straightline_interpolate(v_cst, v_cst, N)
            for k = 1:N
                mix = (k-1)/(N-1)
                x[veh.id_q, k] = vec(slerp_interpolate(traj.q0, traj.qf, mix))
            end
            rot_ang, rot_ax = Log(traj.q0'*traj.qf)
            rot_speed = rot_ang/flight_time
            ω_cst = rot_speed*rot_ax
            x[veh.id_ω, :] = straightline_interpolate(ω_cst, ω_cst, N)

            return x, u, p
        end)

    return nothing
end # function

function set_cost!(pbm::TrajectoryProblem,
                   algo::Symbol)::Nothing

    problem_set_running_cost!(
        pbm, algo,
        (t, k, x, u, p, q, pbm) -> begin
            traj = pbm.mdl.traj
            veh = pbm.mdl.vehicle

            T = u[veh.id_T]
            M = u[veh.id_M]
            f = u[veh.id_rcs]

            T_max_sq = veh.T_max^2
            M_max_sq = veh.M_max^2
            f_max_sq = veh.csm.imp_max^2

            return (T'*T)/T_max_sq+(M'*M)/M_max_sq+(f'*f)/f_max_sq
        end)

    return nothing
end # function

function set_dynamics!(pbm::TrajectoryProblem)::Nothing

    problem_set_dynamics!(
        pbm,
        # Function value f
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            impulse = k<0

            tdil = p[veh.id_t] # Time dilation
            r = x[veh.id_r]
            v = x[veh.id_v]
            q = Quaternion(x[veh.id_q])
            ω = x[veh.id_ω]
            T = u[veh.id_T]
            M = u[veh.id_M]
            rcs = u[veh.id_rcs]

            n_rcs = length(veh.id_rcs)
            dir_rcs = [veh.csm.f_rcs[veh.csm.rcs_select[i]] for i=1:n_rcs]
            iJ = inv(veh.J)
            xi, yi, zi = env.xi, env.yi, env.zi
            norb = env.n

            f = zeros(pbm.nx)
            f[veh.id_v] = T/veh.m
            f[veh.id_v] += sum(rcs[i]*dir_rcs[i] for i=1:n_rcs)/veh.m
            f[veh.id_ω] = iJ*M
            if !impulse
                # Rigid body terms
                f[veh.id_r] = v
                f[veh.id_q] = 0.5*vec(q*ω)
                f[veh.id_ω] += -iJ*cross(ω, veh.J*ω)
                # Clohessy-Wiltshire dynamics terms
                f[veh.id_v] += (-2*norb*dot(zi, v))*xi
                f[veh.id_v] += (-norb^2*dot(yi, r))*yi
                f[veh.id_v] += (3*norb^2*dot(zi, r)+2*norb*dot(xi, v))*zi
                # Scale for absolute time
                f *= tdil
            end

            return f
        end,
        # Jacobian df/dx
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle

            tdil = p[veh.id_t]
            v = x[veh.id_v]
            q = Quaternion(x[veh.id_q])
            ω = x[veh.id_ω]

            dfqdq = 0.5*skew(Quaternion(ω), :R)
            dfqdω = 0.5*skew(q)
            dfωdω = -veh.J\(skew(ω)*veh.J-skew(veh.J*ω))

            A = zeros(pbm.nx, pbm.nx)
            A[veh.id_r, veh.id_v] = I(3)
            A[veh.id_q, veh.id_q] = dfqdq
            A[veh.id_q, veh.id_ω] = dfqdω[:, 1:3]
            A[veh.id_ω, veh.id_ω] = dfωdω
            A *= tdil

            return A
        end,
        # Jacobian df/du
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            impulse = k<0

            tdil = p[veh.id_t]

            n_rcs = length(veh.id_rcs)
            dir_rcs = [veh.csm.f_rcs[veh.csm.rcs_select[i]] for i=1:n_rcs]

            B = zeros(pbm.nx, pbm.nu)
            B[veh.id_v, veh.id_T] = (1.0/veh.m)*I(3)
            for i=1:n_rcs
                B[veh.id_v, veh.id_rcs[i]] = dir_rcs[i]/veh.m
            end
            B[veh.id_ω, veh.id_M] = veh.J\I(3)
            if !impulse
                # Scale for absolute time
                B *= tdil
            end

            return B
        end,
        # Jacobian df/dp
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle

            tdil = p[veh.id_t]

            F = zeros(pbm.nx, pbm.np)
            F[:, veh.id_t] = pbm.f(t, k, x, u, p)/tdil

            return F
        end)

    return nothing
end # function

function set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (t, k, u, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle

            T = u[veh.id_T]
            M = u[veh.id_M]
            f = u[veh.id_rcs]

            @add_constraint(
                ocp, SOC, "thrust_bound",
                (T,), begin
                    local T, = arg #noerr
                    vcat(veh.T_max, T)
                end)

            @add_constraint(
                ocp, SOC, "torque_bound",
                (M,), begin
                    local M, = arg #noerr
                    vcat(veh.M_max, M)
                end)

            @add_constraint(
                ocp, LINF, "rcs_impulse_bound",
                (f,), begin
                    local f, = arg #noerr
                    vcat(veh.csm.imp_max, f)
                end)

        end)

    return nothing
end # function

function set_nonconvex_constraints!(pbm::TrajectoryProblem,
                                    algo::Symbol)::Nothing

    # TODO

    return nothing
end # function

function set_bcs!(pbm::TrajectoryProblem)::Nothing

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
end # function
