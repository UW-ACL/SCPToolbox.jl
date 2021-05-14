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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function define_problem!(pbm::TrajectoryProblem,
                         algo::T_Symbol)::Nothing
    _common__set_dims!(pbm)
    _common__set_scale!(pbm)
    _common__set_integration!(pbm)
    _common__set_cost!(pbm, algo)
    _common__set_convex_constraints!(pbm)
    _common__set_nonconvex_constraints!(pbm, algo)
    _common__set_bcs!(pbm)

    _common__set_guess!(pbm)

    return nothing
end

function _common__set_dims!(pbm::TrajectoryProblem)::Nothing

    # Parameters
    np = pbm.mdl.vehicle.id_δ[end]

    problem_set_dims!(pbm, 13, 6, np)

    return nothing
end

function _common__set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl

    veh, traj = mdl.vehicle, mdl.traj
    min_pos, max_pos = min.(traj.r0, traj.rf), max.(traj.r0, traj.rf)
    for i in veh.id_r
        problem_advise_scale!(pbm, :state, i,
                              (min_pos[i], max_pos[i]))
    end
    problem_advise_scale!(pbm, :parameter, veh.id_t,
                          (traj.tf_min, traj.tf_max))
    for i in veh.id_δ
        problem_advise_scale!(pbm, :parameter, i,
                              (-100.0, 1.0))
    end

    return nothing
end # function

function _common__set_integration!(pbm::TrajectoryProblem)::Nothing

    # Quaternion re-normalization on numerical integration step
    problem_set_integration_action!(
        pbm, pbm.mdl.vehicle.id_q,
        (x, pbm) -> begin
            xn = x/norm(x)
            return xn
        end)

    return nothing
end # function

function _common__set_guess!(pbm::TrajectoryProblem)::Nothing

    # Use an L-shaped axis-aligned position trajectory, a corresponding
    # velocity trajectory, a SLERP interpolation for the quaternion attitude
    # and a corresponding constant-speed angular velocity.

    problem_set_guess!(
        pbm,
        (N, pbm) -> begin

            # Parameters
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj

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
                tk = @k(times)[1]
                for i = 1:3
                    if tk <= flight_time_leg_cumul[i]
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
                    end
                end
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

            # Update room SDF parameter guess
            δ = reshape(view(p, veh.id_δ), env.n_iss, :)
            for i = 1:env.n_iss
                δi = view(δ, i, :)
                roomi = env.iss[i]
                for k = 1:N
                    @k(δi) = 1-norm((@k(r)-roomi.c)./roomi.s, Inf)
                end
            end

            # >> Input guess <<
            idle = zeros(pbm.nu)
            u = straightline_interpolate(idle, idle, N)

            return x, u, p
        end)

    return nothing
end # function

function _common__set_cost!(pbm::TrajectoryProblem,
                            algo::T_Symbol)::Nothing

    # Terminal cost
    problem_set_terminal_cost!(
        pbm, (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            tdil = p[veh.id_t]
            δ = p[veh.id_δ]
            tdil_max = traj.tf_max
            γ = traj.γ
            ε_sdf = traj.ε_sdf
            return γ*(tdil/tdil_max)^2+ε_sdf*sum(-δ)
        end)

    # Running cost
    if algo==:scvx
        problem_set_running_cost!(
            pbm, algo,
            (t, k, x, u, p, pbm) -> begin
                traj = pbm.mdl.traj
                veh = pbm.mdl.vehicle
                T_max_sq = veh.T_max^2
                M_max_sq = veh.M_max^2
                T = u[veh.id_T]
                M = u[veh.id_M]
                γ = traj.γ
                return (1-γ)*((T'*T)/T_max_sq+(M'*M)/M_max_sq)
            end)
    else
        problem_set_running_cost!(
            pbm, algo,
            # Input quadratic penalty S
            (t, k, p, pbm) -> begin
                traj = pbm.mdl.traj
                veh = pbm.mdl.vehicle
                T_max_sq = veh.T_max^2
                M_max_sq = veh.M_max^2
                γ = traj.γ
                S = zeros(pbm.nu, pbm.nu)
                S[veh.id_T, veh.id_T] = (1-γ)*I(3)/T_max_sq
                S[veh.id_M, veh.id_M] = (1-γ)*I(3)/M_max_sq
                return S
            end)
    end

    return nothing
end # function

function _common__set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the state
    problem_set_X!(
        pbm, (t, k, x, p, pbm) -> begin
            traj = pbm.mdl.traj
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env

            room = env.iss

            r = x[veh.id_r]
            v = x[veh.id_v]
            ω = x[veh.id_ω]
            tdil = p[veh.id_t]
            δ = reshape(p[veh.id_δ], env.n_iss, :)

            C = T_ConvexConeConstraint
            X = Array{C}(undef, 4+env.n_iss)
            X[1:4] = [C(vcat(veh.v_max, v), :soc),
                      C(vcat(veh.ω_max, ω), :soc),
                      C(tdil-traj.tf_max, :nonpos),
                      C(traj.tf_min-tdil, :nonpos)]
            # Individual space station room SDFs
            for i = 1:env.n_iss
                X[i+4] = C(vcat(1-@k(δ)[i], (r-room[i].c)./room[i].s), :linf)
            end

            return X
        end)

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (t, k, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle

            T = u[veh.id_T]
            M = u[veh.id_M]

            C = T_ConvexConeConstraint
            U = [C(vcat(veh.T_max, T), :soc),
                 C(vcat(veh.M_max, M), :soc)]

            return U
        end)

    return nothing
end

function _common__set_nonconvex_constraints!(
    pbm::TrajectoryProblem,
    algo::T_Symbol)::Nothing

    # Constraint s
    _ff__s = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj

        r = x[veh.id_r]
        δ = @k(reshape(p[veh.id_δ], env.n_iss, :))

        s = zeros(env.n_obs+1)
        # Ellipsoidal obstacles
        for i = 1:env.n_obs
            E = env.obs[i]
            s[i] = 1-E(r)
        end
        # Space station flight space SDF
        d = logsumexp(δ; t=traj.hom)
        s[end] = -d

        return s
    end

    # Jacobian ds/dx
    _ff__C = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj

        r = x[veh.id_r]

        C = zeros(env.n_obs+1, pbm.nx)
        # Ellipsoidal obstacles
        for i = 1:env.n_obs
            E = env.obs[i]
            C[i, veh.id_r] = -∇(E, r)
        end

        return C
    end

    # Jacobian ds/dp
    _ff__G = (t, k, x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        env = pbm.mdl.env
        traj = pbm.mdl.traj

        room = env.iss
        E = T_RealMatrix(I(env.n_iss))

        r = x[veh.id_r]
        id_δ = @k(reshape(veh.id_δ, env.n_iss, :))
        δ = p[id_δ]

        G = zeros(env.n_obs+1, pbm.np)
        # Space station flight space SDF
        _, ∇d = logsumexp(δ, [E[:, i] for i=1:env.n_iss]; t=traj.hom)
        G[end, id_δ] = -∇d

        return G
    end

    if algo==:scvx
        problem_set_s!(pbm, algo, _ff__s, _ff__C, nothing, _ff__G)
    else
        _ff___s = (t, k, x, p, pbm) -> _ff__s(t, k, x, nothing, p, pbm)
        _ff___C = (t, k, x, p, pbm) -> _ff__C(t, k, x, nothing, p, pbm)
        _ff___G = (t, k, x, p, pbm) -> _ff__G(t, k, x, nothing, p, pbm)
        problem_set_s!(pbm, algo, _ff___s, _ff___C, _ff___G)
    end

end # function

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
end # function
