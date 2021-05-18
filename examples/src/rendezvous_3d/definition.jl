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

using Printf

using Parser

# ..:: Methods ::..

function define_problem!(pbm::TrajectoryProblem,
                         algo::Symbol,
                         N::Int)::Nothing

    set_dims!(pbm)
    set_scale!(pbm)
    set_integration!(pbm)
    set_callback!(pbm)
    set_cost!(pbm, algo)
    set_dynamics!(pbm)
    set_convex_constraints!(pbm, N)
    set_nonconvex_constraints!(pbm, algo)
    set_bcs!(pbm)

    set_guess!(pbm)

    return nothing
end # function

function set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 13, 33, 14)

    return nothing
end # function

function set_scale!(pbm::TrajectoryProblem)::Nothing

    # Parameters
    mdl = pbm.mdl
    veh = mdl.vehicle
    traj = mdl.traj
    env = mdl.env
    n_rcs = length(veh.id_rcs)

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
    advise!(pbm, :input, veh.id_rcs, (0, veh.csm.imp_max))
    advise!(pbm, :input, veh.id_rcs_ref, (0, veh.csm.imp_max))
    advise!(pbm, :input, veh.id_rcs_eq, (0, n_rcs*veh.csm.imp_min))

    # >> Parameters <<
    vf = dot(traj.vf, env.xi)
    advise!(pbm, :parameter, veh.id_t, (traj.tf_min, traj.tf_max))
    advise!(pbm, :parameter, veh.id_dock_tol[veh.id_r],
            (-traj.r_radial_tol, traj.r_radial_tol))
    advise!(pbm, :parameter, veh.id_dock_tol[veh.id_v[1]],
            (-traj.v_axial_max-vf, -traj.v_axial_min-vf))
    advise!(pbm, :parameter, veh.id_dock_tol[veh.id_v[2:3]],
            (-traj.v_radial_tol, traj.v_radial_tol))
    advise!(pbm, :parameter, veh.id_dock_tol[veh.id_ω],
            (-traj.ω_tol, traj.ω_tol))

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

function set_callback!(pbm::TrajectoryProblem)::Nothing

    # Callback to update homotopy parameter
    problem_set_callback!(
        pbm, (bay, subproblem, mdl) -> begin
            pars = subproblem.def.pars
            sol = subproblem.sol
            ref = subproblem.ref
            traj = mdl.traj

            # Save current homotopy
            bay[:κ1] = traj.κ1 # Current homotopy

            # >> Update logic for homotopy value <<

            increase_homotopy = sol.improv_rel<=mdl.traj.β

            i_last = haskey(ref.bay, :last_update) ? ref.bay[:last_update] : 1
            if increase_homotopy
                i = findfirst(mdl.traj.κ1_grid.==mdl.traj.κ1)
                if i<length(mdl.traj.κ1_grid)

                    traj.κ1 = mdl.traj.κ1_grid[i+1]

                    # Update maximum iterations to maintain iter_max for
                    # solving with the new homotopy value
                    pars.iter_max += sol.iter-i_last
                    bay[:last_update] = sol.iter

                else
                    # Homotopy is at maximum value, can't go any higher
                    increase_homotopy = false
                end
            else
                bay[:last_update] = i_last
            end
            bay[:κ1_updated] = increase_homotopy

            return increase_homotopy
        end)

    # Add table column to show homotopy parameter
    problem_add_table_column!(
        pbm, :homotopy_κ1, "κ1", "%s", 10,
        bay->@sprintf("%.2e%s", bay[:κ1], bay[:κ1_updated] ? "*" : ""))

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

            f = u[veh.id_rcs]
            feq = u[veh.id_rcs_eq]

            f_min = veh.csm.imp_min
            f_max = veh.csm.imp_max

            return sum(f)/f_max+traj.γ*feq/f_min
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
            rcs = u[veh.id_rcs]

            n_rcs = length(veh.id_rcs)
            dir_rcs = [veh.csm.f_rcs[veh.csm.rcs_select[i]] for i=1:n_rcs]
            pos_rcs = [veh.csm.r_rcs[veh.csm.rcs_select[i]] for i=1:n_rcs]
            iJ = inv(veh.csm.J)
            xi, yi, zi = env.xi, env.yi, env.zi
            norb = env.n

            f = zeros(pbm.nx)
            f[veh.id_v] = sum(rcs[i]*dir_rcs[i] for i=1:n_rcs)/veh.csm.m
            f[veh.id_ω] = iJ*sum(rcs[i]*cross(pos_rcs[i], dir_rcs[i])
                                 for i=1:n_rcs)
            if !impulse
                # Rigid body terms
                f[veh.id_r] = v
                f[veh.id_q] = 0.5*vec(q*ω)
                f[veh.id_ω] += -iJ*cross(ω, veh.csm.J*ω)
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
            env = pbm.mdl.env

            tdil = p[veh.id_t]
            v = x[veh.id_v]
            q = Quaternion(x[veh.id_q])
            ω = x[veh.id_ω]

            xi, yi, zi = env.xi, env.yi, env.zi
            norb = env.n

            # Rigid body terms
            dfqdq = 0.5*skew(Quaternion(ω), :R)
            dfqdω = 0.5*skew(q)
            dfωdω = -veh.csm.J\(skew(ω)*veh.csm.J-skew(veh.csm.J*ω))

            # Clohessy-Wiltshire dynamics terms
            dfvdv = 2*norb*(zi*xi'-xi*zi')
            dfvdr = norb^2*(3*zi*zi'-yi*yi')

            A = zeros(pbm.nx, pbm.nx)
            A[veh.id_r, veh.id_v] = I(3)
            A[veh.id_v, veh.id_r] = dfvdr
            A[veh.id_v, veh.id_v] = dfvdv
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
            pos_rcs = [veh.csm.r_rcs[veh.csm.rcs_select[i]] for i=1:n_rcs]
            iJ = inv(veh.csm.J)

            B = zeros(pbm.nx, pbm.nu)
            for i=1:n_rcs
                B[veh.id_v, veh.id_rcs[i]] = dir_rcs[i]/veh.csm.m
            end
            for i=1:n_rcs
                B[veh.id_ω, veh.id_rcs[i]] = iJ*cross(pos_rcs[i], dir_rcs[i])
            end
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

function set_convex_constraints!(pbm::TrajectoryProblem,
                                 N::Int)::Nothing

    # Convex path constraints on the state
    problem_set_X!(
        pbm, (t, k, x, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj
            env = pbm.mdl.env

            if k==N

                qf = x[veh.id_q]
                Δxf = p[veh.id_dock_tol]
                Δrf = Δxf[veh.id_r]
                Δvf = Δxf[veh.id_v]
                Δωf = Δxf[veh.id_ω]

                xi = env.xi

                @add_constraint(
                    ocp, LINF, "dock_pos_tol",
                    (Δrf,), begin
                        local Δrf, = arg #noerr
                        vcat(0.1, Δrf)
                    end)

                @add_constraint(
                    ocp, ZERO, "dock_pos_axial_exact",
                    (Δrf,), begin
                        local Δrf, = arg #noerr
                        dot(Δrf, xi)
                    end)

                @add_constraint(
                    ocp, LINF, "dock_vel_tol",
                    (Δvf,), begin
                        local Δvf, = arg #noerr
                        vcat(0.01, Δvf)
                    end)

                ang_max = deg2rad(1)
                @add_constraint(
                    ocp, NONPOS, "dock_att_tol",
                    (qf,), begin
                        local qf = arg[1] #noerr
                        qerr_w = qf[1:3]'*traj.qf.v+qf[4]*traj.qf.w
                        cos(ang_max/2)-qerr_w
                    end)

                @add_constraint(
                    ocp, LINF, "dock_ang_vel_tol",
                    (Δωf,), begin
                        local Δωf, = arg #noerr
                        vcat(deg2rad(0.01), Δωf)
                    end)

            end

        end)

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (t, k, u, p, pbm, ocp) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            f = u[veh.id_rcs]
            fr = u[veh.id_rcs_ref]
            feq = u[veh.id_rcs_eq]
            tdil = p[veh.id_t]

            @add_constraint(
                ocp, NONPOS, "rcs_impulse_nonneg",
                (f,), begin
                    local f, = arg #noerr
                    -f
                end)

            @add_constraint(
                ocp, NONPOS, "rcs_impulse_ref_nonneg",
                (fr,), begin
                    local fr, = arg #noerr
                    -fr
                end)

            @add_constraint(
                ocp, LINF, "rcs_impulse_max",
                (f,), begin
                    local f, = arg #noerr
                    vcat(veh.csm.imp_max, f)
                end)

            @add_constraint(
                ocp, LINF, "rcs_impulse_ref_max",
                (fr,), begin
                    local fr, = arg #noerr
                    vcat(veh.csm.imp_max, fr)
                end)

            @add_constraint(
                ocp, L1, "rcs_impulse_ref_equality",
                (f, fr, feq,), begin
                    local f, fr, feq = arg #noerr
                    vcat(feq, f-fr)
                end)

            @add_constraint(
                ocp, NONPOS, "min_time",
                (tdil, ), begin
                    local tdil, = arg #noerr
                    tdil[1]-traj.tf_max
                end)

            @add_constraint(
                ocp, NONPOS, "max_time",
                (tdil, ), begin
                    local tdil, = arg #noerr
                    traj.tf_min-tdil[1]
                end)

        end)

    return nothing
end # function

function set_nonconvex_constraints!(pbm::TrajectoryProblem,
                                    algo::Symbol)::Nothing

    # Parameters
    veh = pbm.mdl.vehicle
    n_rcs = length(veh.id_rcs)
    _common_s_sz = 2*n_rcs

    problem_set_s!(
        pbm, algo,
        # Constraint s
        (t, k, x, u, p, pbm) -> begin
            veh = pbm.mdl.vehicle
            traj = pbm.mdl.traj

            s = zeros(_common_s_sz)

            for i=1:n_rcs
                id_f, id_fr = veh.id_rcs[i], veh.id_rcs_ref[i]
                f, fr = u[id_f], u[id_fr]
                above_mib = fr-veh.csm.imp_min
                OR = or(above_mib;
                        κ1=traj.κ1, κ2=traj.κ2,
                        # minval=-veh.csm.imp_min,
                        maxval=veh.csm.imp_max-veh.csm.imp_min)
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
                id_f, id_fr = veh.id_rcs[i], veh.id_rcs_ref[i]
                fr = u[id_fr]
                above_mib = fr-veh.csm.imp_min
                ∇above_mib = [1.0]
                OR, ∇OR = or((above_mib, ∇above_mib);
                             κ1=traj.κ1, κ2=traj.κ2,
                             # minval=-veh.csm.imp_min,
                             maxval=veh.csm.imp_max-veh.csm.imp_min)
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
            Δx = p[veh.id_dock_tol]
            g = x+Δx-rhs
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
            K[:, veh.id_dock_tol] = I(pbm.nx)
            return K
        end)

    return nothing
end # function
