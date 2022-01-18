"""
Quadrotor obstacle avoidance problem definition.

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
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

using JuMP
using ECOS
using Printf

# ..:: Methods ::..

function define_problem!(
        pbm::TrajectoryProblem,
        algo::Symbol
)::Nothing
    set_dims!(pbm)
    set_scale!(pbm)
    set_cost!(pbm, algo)
    set_dynamics!(pbm)
    set_convex_constraints!(pbm, algo)
    set_nonconvex_constraints!(pbm, algo)
    set_bcs!(pbm)

    set_guess!(pbm)

    return nothing
end

function set_dims!(
        pbm::TrajectoryProblem
)::Nothing

    problem_set_dims!(pbm, 6, 4, 1)

    return nothing
end

function set_scale!(
        pbm::TrajectoryProblem
)::Nothing

    mdl = pbm.mdl

    tdil_min = mdl.traj.tf_min
    tdil_max = mdl.traj.tf_max
    tdil_max_adj = tdil_min+1.0*(tdil_max-tdil_min)
    problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t,
                          (tdil_min, tdil_max_adj))

    return nothing
end

function set_guess!(
        pbm::TrajectoryProblem
)::Nothing

    problem_set_guess!(
        pbm, (N, pbm) ->
        begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            g = pbm.mdl.env.g

            # Parameter guess
            p = zeros(pbm.np)
            p[veh.id_t] = 0.5*(traj.tf_min+traj.tf_max)

            # State guess
            x0 = zeros(pbm.nx)
            xf = zeros(pbm.nx)
            x0[veh.id_r] = traj.r0
            xf[veh.id_r] = traj.rf
            x0[veh.id_v] = traj.v0
            xf[veh.id_v] = traj.vf
            x = straightline_interpolate(x0, xf, N)

            # Input guess
            hover = zeros(pbm.nu)
            hover[veh.id_u] = -g
            hover[veh.id_σ] = norm(g)
            u = straightline_interpolate(hover, hover, N)

            return x, u, p
        end)

    return nothing
end

function set_cost!(
        pbm::TrajectoryProblem,
        algo::Symbol
)::Nothing

    problem_set_terminal_cost!(
        pbm, (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            tdil = p[veh.id_t]
            tdil_max = traj.tf_max
            γ = traj.γ
            return γ*(tdil/tdil_max)^2
        end)

    # Running cost
    if algo==:scvx
        problem_set_running_cost!(
            pbm, algo,
            (t, k, x, u, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                env = pbm.mdl.env
                traj = pbm.mdl.traj
                σ = u[veh.id_σ]
                hover = norm(env.g)
                γ = traj.γ
                return (1-γ)*(σ/hover)^2
            end)
    else
        problem_set_running_cost!(
            pbm, algo,
            # Input quadratic penalty S
            (t, k, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                env = pbm.mdl.env
                traj = pbm.mdl.traj
                hover = norm(env.g)
                γ = traj.γ
                S = zeros(pbm.nu, pbm.nu)
                S[veh.id_σ, veh.id_σ] = (1-γ)*1/hover^2
                return S
            end)
    end


    return nothing
end

function set_dynamics!(
        pbm::TrajectoryProblem
)::Nothing

    problem_set_dynamics!(
        pbm,
        # Dynamics f
        (t, k, x, u, p, pbm) -> begin
            g = pbm.mdl.env.g
            veh = pbm.mdl.vehicle
            v = x[veh.id_v]
            uu = u[veh.id_u]
            tdil = p[veh.id_t]
            f = zeros(pbm.nx)
            f[veh.id_r] = v
            f[veh.id_v] = uu+g
            f *= tdil
            return f
        end,
        # Jacobian df/dx
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            tdil = p[veh.id_t]
            A = zeros(pbm.nx, pbm.nx)
            A[veh.id_r, veh.id_v] = I(3)
            A *= tdil
            return A
        end,
        # Jacobian df/du
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            tdil = p[veh.id_t]
            B = zeros(pbm.nx, pbm.nu)
            B[veh.id_v, veh.id_u] = I(3)
            B *= tdil
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
end

function set_convex_constraints!(
        pbm::TrajectoryProblem,
        algo::Symbol
)::Nothing

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (t, k, u, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            common = (pbm, ocp, algo)

            a = u[veh.id_u]
            σ = u[veh.id_σ]
            tdil = p[veh.id_t]

            define_conic_constraint!(
                common..., NONPOS, "min_accel", (σ,),
                (σ) -> veh.u_min-σ)

            define_conic_constraint!(
                common..., NONPOS, "max_accel", (σ,),
                (σ) -> σ-veh.u_max)

            define_conic_constraint!(
                common..., SOC, "lcvx_equality", (σ, a),
                (σ, a) -> vcat(σ, a))

            define_conic_constraint!(
                common..., NONPOS, "max_tilt", (σ, a),
                (σ, a) -> σ*cos(veh.tilt_max)-a[3])

            define_conic_constraint!(
                common..., NONPOS, "max_duration", (tdil,),
                (tdil) -> tdil-traj.tf_max)

            define_conic_constraint!(
                common..., NONPOS, "min_duration", (tdil,),
                (tdil) -> traj.tf_min-tdil)
        end)

    return nothing
end

function set_nonconvex_constraints!(
        pbm::TrajectoryProblem,
        algo::Symbol
)::Nothing

    # Constraint s
    _q__s = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        s = zeros(env.n_obs)
        for i = 1:env.n_obs
            E = env.obs[i]
            r = x[veh.id_r]
            s[i] = 1-E(r)
        end
        return s
    end

    # Jacobian ds/dx
    _q__C = (t, k, x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        C = zeros(env.n_obs, pbm.nx)
        for i = 1:env.n_obs
            E = env.obs[i]
            r = x[veh.id_r]
            C[i, veh.id_r] = -∇(E, r)
        end
        return C
    end

    if algo==:scvx
        problem_set_s!(pbm, algo, _q__s, _q__C)
    else
        _q___s = (t, k, x, p, pbm) -> _q__s(t, k, x, nothing, p, pbm)
        _q___C = (t, k, x, p, pbm) -> _q__C(t, k, x, nothing, p, pbm)
        problem_set_s!(pbm, algo, _q___s, _q___C)
    end

end

function set_bcs!(
        pbm::TrajectoryProblem
)::Nothing

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
