#= Starship landing flip maneuver example, common code.

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

include("../../models/starship.jl")
include("../../core/problem.jl")
include("../../utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

function define_problem!(pbm::TrajectoryProblem,
                         algo::T_Symbol)::Nothing
    _common__set_dims!(pbm)
    _common__set_scale!(pbm)
    _common__set_cost!(pbm)
    _common__set_dynamics!(pbm)
    _common__set_convex_constraints!(pbm)
    _common__set_nonconvex_constraints!(pbm, algo)
    _common__set_bcs!(pbm)

    _common__set_guess!(pbm)

    return nothing
end

function _common__set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 8, 3, 10)

    return nothing
end

function _common__set_scale!(pbm::TrajectoryProblem)::Nothing

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
    advise!(pbm, :state, veh.id_m, (veh.m-1e3, veh.m))
    advise!(pbm, :state, veh.id_δd, (-veh.δ_max, veh.δ_max))
    # Inputs
    advise!(pbm, :input, veh.id_T, (veh.T_min1, veh.T_max3))
    advise!(pbm, :input, veh.id_δ, (-veh.δ_max, veh.δ_max))
    advise!(pbm, :input, veh.id_δdot, (-veh.δdot_max, veh.δdot_max))
    # Parameters
    advise!(pbm, :parameter, veh.id_t1, (0.0, traj.tf_max))
    advise!(pbm, :parameter, veh.id_t2, (0.0, traj.tf_max))
    for i=1:pbm.nx
        advise!(pbm, :parameter, veh.id_xs[i], pbm.xrg[i])
    end

    return nothing
end

function _common__set_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(pbm, starship_initial_guess)

    return nothing
end

function _common__set_cost!(pbm::TrajectoryProblem)::Nothing

    problem_set_terminal_cost!(pbm, (x, p, pbm) -> begin
                               veh = pbm.mdl.vehicle
                               traj = pbm.mdl.traj
                               env = pbm.mdl.env
                               # Altitude at phase switch
                               # Goal: maximize it
                               rs = p[veh.id_xs][veh.id_r]
                               alt = dot(rs, env.ey)
                               alt_nrml = traj.hs
                               alt_cost = -alt/alt_nrml
                               μ = 0.3 # Relative weight to fuel cost
                               # Fuel consumption
                               # Goal: minimize it
                               mf = x[veh.id_m]
                               Δm = 0.0-mf
                               Δm_nrml = 10e3
                               Δm_cost = Δm/Δm_nrml
                               # Total cost
                               return μ*alt_cost+Δm_cost
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
        env = pbm.mdl.env
        traj = pbm.mdl.traj
        v = x[veh.id_v]
        θ = x[veh.id_θ]
        m = x[veh.id_m]
        T = u[veh.id_T]
        δ = u[veh.id_δ]
        tdil = ((t<=traj.τs) ? p[veh.id_t1]/traj.τs :
                p[veh.id_t2]/(1-traj.τs))

        ℓcp = veh.lcp-veh.lcg
        ei = veh.ei(θ)
        ej = veh.ej(θ)
        D = -veh.CD*norm(v)*v
        ∇θ_Tv = T*(-sin(δ)*ej+cos(δ)*-ei)
        ∇v_D = -veh.CD*(norm(v)*I(2)+(v*v')/norm(v))
        ∇v_MD = -ℓcp*∇v_D'*ei
        ∇θ_MD = -ℓcp*dot(D, ej)

        A = zeros(pbm.nx, pbm.nx)
        A[veh.id_r, veh.id_v] = I(2)
        A[veh.id_v, veh.id_v] = (∇v_D)/veh.m
        A[veh.id_v, veh.id_θ] = (∇θ_Tv)/veh.m
        A[veh.id_θ, veh.id_ω] = 1.0
        A[veh.id_ω, veh.id_v] = ∇v_MD/veh.J
        A[veh.id_ω, veh.id_θ] = ∇θ_MD/veh.J
        A[veh.id_δd, veh.id_δd] = -1.0/veh.rate_delay

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
        tdil = ((t<=traj.τs) ? p[veh.id_t1]/traj.τs :
                p[veh.id_t2]/(1-traj.τs))

        ℓeng = -veh.lcg
        ei = veh.ei(θ)
        ej = veh.ej(θ)
        ∇T_Tv = -sin(δ)*ei+cos(δ)*ej
        ∇δ_Tv = T*(-cos(δ)*ei-sin(δ)*ej)
        ∇T_MT = ℓeng*sin(δ)
        ∇δ_MT = ℓeng*T*cos(δ)

        B = zeros(pbm.nx, pbm.nu)
        B[veh.id_v, veh.id_T] = (∇T_Tv)/veh.m
        B[veh.id_v, veh.id_δ] = (∇δ_Tv)/veh.m
        B[veh.id_ω, veh.id_T] = (∇T_MT)/veh.J
        B[veh.id_ω, veh.id_δ] = (∇δ_MT)/veh.J
        B[veh.id_m, veh.id_T] = veh.αe
        B[veh.id_δd, veh.id_δ] = 1.0/veh.rate_delay

        B *= tdil
        return B
        end,
        # Jacobian df/dp
        (t, k, x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        id_t = (t<=traj.τs) ? veh.id_t1 : veh.id_t2
        F = zeros(pbm.nx, pbm.np)
        F[:, id_t] = pbm.f(t, k, x, u, p)/p[id_t]
        return F
        end)

    return nothing
end

function _common__set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the state
    problem_set_X!(
        pbm, (t, k, x, p, pbm) -> begin
        traj = pbm.mdl.traj
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        r = x[veh.id_r]
        v = x[veh.id_v]
        tf = p[veh.id_t1]+p[veh.id_t2]
        C = T_ConvexConeConstraint
        X = [C(dot(v, env.ey), :nonpos),
             C(tf-traj.tf_max, :nonpos),
             C(traj.tf_min-tf, :nonpos)]
        return X
        end)

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (t, k, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        T = u[veh.id_T]
        δ = u[veh.id_δ]
        flip_phase = t<=traj.τs
        T_max = (flip_phase) ? veh.T_max3 : veh.T_max1
        T_min = (flip_phase) ? veh.T_min3 : veh.T_min1
        C = T_ConvexConeConstraint
        U = [C(T-T_max, :nonpos),
             C(T_min-T, :nonpos),
             C(vcat(veh.δ_max, δ), :l1)]
        return U
        end)

    return nothing
end

function _common__set_nonconvex_constraints!(pbm::TrajectoryProblem,
                                             algo::T_Symbol)::Nothing

    # Return true if this is the temporal node where phase 1 ends
    _common__phase_switch = (t, pbm) -> begin
        Δt = 1/(pbm.scp.N-1)
        τs = pbm.mdl.traj.τs
        tol = 1e-3
        phase_switch = (τs-Δt)+tol<=t && t<=τs+tol
        return phase_switch
    end

    # Return true if this is a phase 2 temporal node
    _common__phase2 = (t, pbm) -> begin
        τs = pbm.mdl.traj.τs
        phase2 = _common__phase_switch(t, pbm) || t>τs
        return phase2
    end

    _common_s_sz = 7+2*pbm.nx

    problem_set_s!(
        pbm, algo,
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
        s[1] = (δ-δd)-δdot*veh.rate_delay
        s[2] = δdot*veh.rate_delay-(δ-δd)
        s[3] = δdot-veh.δdot_max
        s[4] = -veh.δdot_max-δdot
        s[5] = norm(r)*cos(traj.γ_gs)-dot(r, env.ey)
        if _common__phase_switch(t, pbm)
        s[(1:pbm.nx).+5] = p[veh.id_xs]-x
        s[(1:pbm.nx).+(5+pbm.nx)] = x-p[veh.id_xs]
        end
        if _common__phase2(t, pbm)
        s[end-1] = θ-traj.θmax2
        s[end] = -traj.θmax2-θ
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
        ∇nrm_r = (nrm_r<sqrt(eps())) ? zeros(2) : r/nrm_r

        C = zeros(_common_s_sz, pbm.nx)
        C[1, veh.id_δd] = -1.0
        C[2, veh.id_δd] = 1.0
        C[5, veh.id_r] = ∇nrm_r*cos(traj.γ_gs)-env.ey
        if _common__phase_switch(t, pbm)
        C[(1:pbm.nx).+5, :] = -I(pbm.nx)
        C[(1:pbm.nx).+(5+pbm.nx), :] = I(pbm.nx)
        end
        if _common__phase2(t, pbm)
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
        if _common__phase_switch(t, pbm)
        G[(1:pbm.nx).+5, veh.id_xs] = I(pbm.nx)
        G[(1:pbm.nx).+(5+pbm.nx), veh.id_xs] = -I(pbm.nx)
        end

        return G
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
        rhs = zeros(7)
        rhs[1:2] = traj.r0
        rhs[3:4] = traj.v0
        rhs[5] = traj.θ0
        rhs[6] = 0.0
        rhs[7] = 0.0
        g = x[vcat(veh.id_r,
                   veh.id_v,
                   veh.id_θ,
                   veh.id_ω,
                   veh.id_m)]-rhs
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
        end)

    # Terminal conditions
    problem_set_bc!(
        pbm, :tc,
        # Constraint g
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        rhs = zeros(6)
        rhs[1:2] = zeros(2)
        rhs[3:4] = traj.vf
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
