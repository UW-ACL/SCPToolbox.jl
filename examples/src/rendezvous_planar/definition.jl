#= Planar spacecraft rendezvous example, common code.

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
    include("../../../utils/src/trajectory.jl")
    using .Parser
    import .Parser.ConicLinearProgram: @add_constraint
    import .Parser.ConicLinearProgram: ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
end

using LinearAlgebra
using Parser

# ..:: Globals ::..

const Trajectory = T.ContinuousTimeTrajectory
const CLP = ConicLinearProgram

# ..:: Methods ::..

function define_problem!(pbm::TrajectoryProblem,
                         algo::Symbol)::Nothing
    set_dims!(pbm)
    set_scale!(pbm)
    set_cost!(pbm, algo)
    set_dynamics!(pbm)
    set_convex_constraints!(pbm)
    set_nonconvex_constraints!(pbm, algo)
    set_bcs!(pbm)

    set_guess!(pbm)

    return nothing
end # function

function set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 6, 12, 1)

    return nothing
end # function

function set_scale!(pbm::TrajectoryProblem)::Nothing

    # Parameters
    mdl = pbm.mdl
    veh = mdl.vehicle
    traj = mdl.traj
    env = mdl.env

    # Derived quantities
    rx0, ry0 = dot(traj.r0, env.xh), dot(traj.r0, env.yh)
    vx0, vy0 = dot(traj.v0, env.xh), dot(traj.v0, env.yh)
    rx_min = 0.0
    rx_max = max([rx0, 1.0]...)
    ry_min = min([ry0, -0.1]...)
    ry_max = max([ry0, 0.1]...)
    vx_min = min([vx0, -rx0/traj.tf_min, -0.1]...)
    vx_max = min([vx0, 0.1]...)
    vy_min = min([vy0, -ry0/traj.tf_min, -0.1]...)
    vy_max = max([vy0, -ry0/traj.tf_min, 0.1]...)
    θ_min = min([traj.θ0, deg2rad(-1.0)]...)
    θ_max = max([traj.θ0, deg2rad(1.0)]...)
    ω_min = min([-traj.θ0/traj.tf_min, traj.ω0, deg2rad(-1.0)]...)
    ω_max = max([-traj.θ0/traj.tf_min, traj.ω0, deg2rad(1.0)]...)

    advise! = problem_advise_scale!

    # States
    advise!(pbm, :state, veh.id_r[1], (rx_min, rx_max))
    advise!(pbm, :state, veh.id_r[2], (ry_min, ry_max))
    advise!(pbm, :state, veh.id_v[1], (vx_min, vx_max))
    advise!(pbm, :state, veh.id_v[2], (vy_min, vy_max))
    advise!(pbm, :state, veh.id_θ, (θ_min, θ_max))
    advise!(pbm, :state, veh.id_ω, (ω_min, ω_max))
    # Inputs
    for id in [veh.id_f, veh.id_fr]
        for i in id
            advise!(pbm, :input, i, (-veh.f_max, veh.f_max))
        end
    end
    for i in veh.id_l1f
        advise!(pbm, :input, i, (0.0, veh.f_max))
    end
    for i in veh.id_l1feq
        advise!(pbm, :input, i, (0.0, 2*veh.f_max))
    end
    # Parameters
    advise!(pbm, :parameter, veh.id_t, (traj.tf_min, traj.tf_max))

    return nothing
end

function set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(
        pbm, (N, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            env = pbm.mdl.env

            # Parameter guess
            p = zeros(pbm.np)
            p[veh.id_t] = 0.5*(traj.tf_min+traj.tf_max)

            # State guess
            x0 = zeros(pbm.nx)
            xf = zeros(pbm.nx)
            x0[veh.id_r] = traj.r0
            x0[veh.id_v] = -traj.r0/p[veh.id_t]
            x0[veh.id_θ] = traj.θ0
            x0[veh.id_ω] = -traj.θ0/p[veh.id_t]
            xf[veh.id_v] = x0[veh.id_v]
            xf[veh.id_ω] = x0[veh.id_ω]
            x = straightline_interpolate(x0, xf, N)

            # Input guess
            idle = zeros(pbm.nu)
            u = straightline_interpolate(idle, idle, N)

            return x, u, p
        end)

    return nothing
end

function set_cost!(pbm::TrajectoryProblem,
                            algo::Symbol)::Nothing

    problem_set_running_cost!(
        pbm, algo,
        (t, k, x, u, p, q, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            l1f = u[veh.id_l1f]
            l1feq = u[veh.id_l1feq]
            f = u[veh.id_f]
            f_nrml = veh.f_max
            runn = sum(l1f)/f_nrml
            runn += traj.γ*sum(l1feq)/f_nrml
            # f_nrml = veh.f_max^2
            # runn = sum(f.^2)/f_nrml
            return runn
        end)

end

function set_dynamics!(pbm::TrajectoryProblem)::Nothing

    problem_set_dynamics!(
        pbm,
        # Dynamics f
        (t, k, x, u, p, pbm) -> begin
            # Parameters
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            impulse = k<0
            # Current (x, u, p) values
            r = x[veh.id_r]
            v = x[veh.id_v]
            θ = x[veh.id_θ]
            ω = x[veh.id_ω]
            fm, fp, f0 = u[veh.id_f]
            tdil = p[veh.id_t]
            # Derived quantities
            uh = veh.uh(θ)
            vh = veh.vh(θ)
            xh, yh, n = env.xh, env.yh, env.n
            # The dynamics
            f = zeros(pbm.nx)
            f[veh.id_v] = ((fm+fp)*uh+f0*vh)/veh.m
            f[veh.id_ω] = ((fp-fm)*veh.lv-f0*veh.lu)/veh.J
            if !impulse
                f[veh.id_r] = v
                f[veh.id_v] += (2*n*dot(yh, v))*xh
                f[veh.id_v] += (3*n^2*dot(yh, r)-2*n*dot(xh, v))*yh
                f[veh.id_θ] = ω
                # Scale for absolute time
                f *= tdil
            end
            return f
        end,
        # Jacobian df/dx
        (t, k, x, u, p, pbm) -> begin
            # Parameters
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            # Current (x, u, p) values
            θ = x[veh.id_θ]
            fm, fp, f0 = u[veh.id_f]
            tdil = p[veh.id_t]
            # Derived quantities
            ∇θ_uh = -veh.vh(θ)
            ∇θ_vh = veh.uh(θ)
            xh, yh, n = env.xh, env.yh, env.n
            # The Jacobian
            A = zeros(pbm.nx, pbm.nx)
            A[veh.id_r, veh.id_v] = I(2)
            A[veh.id_v, veh.id_r] = 3*n^2*yh*yh'
            A[veh.id_v, veh.id_v] = 2*n*(xh*yh'-yh*xh')
            A[veh.id_v, veh.id_θ] = ((fm+fp)*∇θ_uh+f0*∇θ_vh)/veh.m
            A[veh.id_θ, veh.id_ω] = 1.0
            # Scale for absolute time
            A *= tdil
            return A
        end,
        # Jacobian df/du
        (t, k, x, u, p, pbm) -> begin
            # Parameters
            veh = pbm.mdl.vehicle
            impulse = k<0
            # Current (x, u, p) values
            θ = x[veh.id_θ]
            tdil = p[veh.id_t]
            # Derived quantities
            uh = veh.uh(θ)
            vh = veh.vh(θ)
            id_fm, id_fp, id_f0 = veh.id_f
            # The Jacobian
            B = zeros(pbm.nx, pbm.nu)
            B[veh.id_v, id_fm] = uh/veh.m
            B[veh.id_v, id_fp] = uh/veh.m
            B[veh.id_v, id_f0] = vh/veh.m
            B[veh.id_ω, id_fm] = -veh.lv/veh.J
            B[veh.id_ω, id_fp] = veh.lv/veh.J
            B[veh.id_ω, id_f0] = -veh.lu/veh.J
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
end

function set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (t, k, u, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            n_rcs = length(veh.id_f)

            tdil = p[veh.id_t]

            for i = 1:n_rcs

                f_i = u[veh.id_f[i]]
                fr_i = u[veh.id_fr[i]]
                l1f_i = u[veh.id_l1f[i]]
                l1feq_i = u[veh.id_l1feq[i]]

                @add_constraint(
                    ocp, NONPOS, "thrust_absval_max",
                    (l1f_i,), begin
                        local l1f, = arg #noerr
                        l1f[1]-veh.f_max
                    end)

                @add_constraint(
                    ocp, NONPOS, "thrust_refval_max",
                    (fr_i,), begin
                        local fr, = arg #noerr
                        fr[1]-veh.f_max
                    end)

                @add_constraint(
                    ocp, NONPOS, "thrust_refval_min",
                    (fr_i,), begin
                        local fr, = arg #noerr
                        -fr[1]-veh.f_max
                    end)

                @add_constraint(
                    ocp, L1, "thrust_absval",
                    (l1f_i, f_i), begin
                        local l1f, f = arg #noerr
                        vcat(l1f, f)
                    end)

                @add_constraint(
                    ocp, L1, "thrust_absval",
                    (l1feq_i, f_i, fr_i), begin
                        local l1feq, f, fr = arg #noerr
                        vcat(l1feq, f-fr)
                    end)

            end

            @add_constraint(
                ocp, NONPOS, "min_time_bound",
                (tdil, ), begin
                    local tdil, = arg #noerr
                    tdil[1]-traj.tf_max
                end)

            @add_constraint(
                ocp, NONPOS, "max_time_bound",
                (tdil, ), begin
                    local tdil, = arg #noerr
                    traj.tf_min-tdil[1]
                end)

            return nothing
        end)

    return nothing
end

function set_nonconvex_constraints!(pbm::TrajectoryProblem,
                                             algo::Symbol)::Nothing

    veh = pbm.mdl.vehicle
    n_rcs = length(veh.id_f)
    _common_s_sz = 2*n_rcs

    problem_set_s!(
        pbm, algo,
        # Constraint s
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            s = zeros(_common_s_sz)

            for i=1:n_rcs
                id_f, id_fr = veh.id_f[i], veh.id_fr[i]
                f, fr = u[id_f], u[id_fr]
                above_db = fr-veh.f_db
                below_db = -veh.f_db-fr
                OR = or(above_db, below_db;
                        κ1=traj.κ1, κ2=traj.κ2,
                        minval=-veh.f_max-veh.f_db,
                        maxval=veh.f_max+veh.f_db)
                s[2*(i-1)+1] = f-OR*fr
                s[2*(i-1)+2] = OR*fr-f
            end

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

            D = zeros(_common_s_sz, pbm.nu)

            for i = 1:n_rcs
                id_f, id_fr = veh.id_f[i], veh.id_fr[i]
                fr = u[id_fr]
                above_db = fr-veh.f_db
                ∇above_db = [1.0]
                below_db = -veh.f_db-fr
                ∇below_db = [-1.0]
                OR, ∇OR = or((above_db, ∇above_db),
                             (below_db, ∇below_db);
                             κ1=traj.κ1, κ2=traj.κ2,
                             minval=-veh.f_max-veh.f_db,
                             maxval=veh.f_max+veh.f_db)
                ∇ORfr = ∇OR[1]*fr+OR
                D[2*(i-1)+1, id_f] = 1.0
                D[2*(i-1)+1, id_fr] = -∇ORfr
                D[2*(i-1)+2, id_f] = -1.0
                D[2*(i-1)+2, id_fr] = ∇ORfr
            end

            return D
        end,
        # Jacobian ds/dp
        (t, k, x, u, p, pbm) -> begin
            G = zeros(_common_s_sz, pbm.np)
            return G
        end)

    return nothing
end

function set_bcs!(pbm::TrajectoryProblem)::Nothing

    # Initial conditions
    problem_set_bc!(
        pbm, :ic,
        # Constraint g
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            rhs = zeros(6)
            rhs[1:2] = traj.r0
            rhs[3:4] = traj.v0
            rhs[5] = traj.θ0
            rhs[6] = traj.ω0
            g = x[vcat(veh.id_r,
                       veh.id_v,
                       veh.id_θ,
                       veh.id_ω)]-rhs
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
        end)

    # Initial conditions
    problem_set_bc!(
        pbm, :tc,
        # Constraint g
        (x, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            env = pbm.mdl.env
            traj = pbm.mdl.traj
            rhs = zeros(6)
            rhs[1:2] = zeros(2)
            rhs[3:4] = -traj.vf*env.xh
            rhs[5] = 0.0
            rhs[6] = 0.0
            g = x[vcat(veh.id_r,
                       veh.id_v,
                       veh.id_θ,
                       veh.id_ω)]-rhs
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
        end)

    return nothing
end
