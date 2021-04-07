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

function define_problem!(pbm::TrajectoryProblem)::Nothing
    _common__set_dims!(pbm)
    _common__set_scale!(pbm)
    _common__set_cost!(pbm)
    _common__set_dynamics!(pbm)
    _common__set_convex_constraints!(pbm)
    _common__set_nonconvex_constraints!(pbm)
    _common__set_bcs!(pbm)

    _common__set_guess!(pbm)

    return nothing
end

function _common__set_dims!(pbm::TrajectoryProblem)::Nothing

    problem_set_dims!(pbm, 9, 3, 2)

    return nothing
end

function _common__set_scale!(pbm::TrajectoryProblem)::Nothing

    mdl = pbm.mdl
    # Parameters
    problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t1,
                          (0.0, mdl.traj.tf_max))
    problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t2,
                          (0.0, mdl.traj.tf_max))
    # Inputs
    problem_advise_scale!(pbm, :input, mdl.vehicle.id_T,
                          (mdl.vehicle.T_min1, mdl.vehicle.T_max3))
    problem_advise_scale!(pbm, :input, mdl.vehicle.id_δ,
                          (-mdl.vehicle.δ_max, mdl.vehicle.δ_max))
    problem_advise_scale!(pbm, :input, mdl.vehicle.id_δdot,
                          (-mdl.vehicle.δdot_max, mdl.vehicle.δdot_max))
    # States
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_r[1],
                          (-100.0, 100.0))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_r[2],
                          (0.0, mdl.traj.r0[2]))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_v[1],
                          (-10.0, 10.0))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_v[2],
                          (mdl.traj.v0[2], 0.0))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_θ,
                          (0.0, mdl.traj.θ0))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_ω,
                          (-deg2rad(10.0), deg2rad(10.0)))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_m,
                          (mdl.vehicle.m-1e3, mdl.vehicle.m))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_δd,
                          (-mdl.vehicle.δ_max, mdl.vehicle.δ_max))
    problem_advise_scale!(pbm, :state, mdl.vehicle.id_τ,
                          (0.0, 1.0))

    return nothing
end

function _common__set_guess!(pbm::TrajectoryProblem)::Nothing

    # Parameters
    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj
    env = pbm.mdl.env

    # Initial condition
    X0 = zeros(pbm.nx)
    X0[veh.id_r] = traj.r0
    X0[veh.id_v] = traj.v0
    X0[veh.id_θ] = traj.θ0
    X0[veh.id_δd] = veh.δ_max

    # Simple guess control strategy
    # Gimbal bang-bang drive θ0 to θs at min 3-engine thrust
    _startship__ac = veh.lcg/veh.J*veh.T_min3*sin(veh.δ_max)
    _startship__ts = sqrt((traj.θ0-traj.θs)/_startship__ac)
    _startship__control = (t, pbm) -> begin
        veh = pbm.mdl.vehicle
        T = veh.T_min3
        ts = _startship__ts
        if t<=ts
            δ = veh.δ_max
        elseif t>ts && t<=2*ts
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
    _startship__f_guess = (t, x, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        u = _startship__control(t, pbm)
        p = zeros(pbm.np)
        p[veh.id_t1] = traj.τs
        p[veh.id_t2] = 1-traj.τs
        dxdt = dynamics(x, u, p, pbm; no_aero_torques=true)
        return dxdt
    end

    problem_set_guess!(pbm, (N, pbm) -> begin

                       # Parameters
                       veh = pbm.mdl.vehicle
                       traj = pbm.mdl.traj
                       env = pbm.mdl.env

                       # Normalized time grid
                       τ_grid = LinRange(0.0, 1.0, N)
                       τs = traj.τs
                       id_phase1 = τ_grid.<=τs
                       id_phase2 = τ_grid.>τs

                       # Initialize empty trajectory guess
                       x = T_RealMatrix(undef, pbm.nx, N)
                       u = T_RealMatrix(undef, pbm.nu, N)

                       # >>>>>>>>><<<<<<<<<<
                       # >> Phase 1: flip <<
                       # >>>>>>>>><<<<<<<<<<

                       # The flip guess dynamics
                       ts = _startship__ts
                       f = _startship__f_guess
                       ctrl = _startship__control

                       # Propagate the flip dynamics under the guess control
                       t_θcst = 10.0
                       tf = 2*ts+t_θcst
                       t = T_RealVector(LinRange(0.0, tf, 5000))
                       X = rk4((t, x) -> f(t, x, pbm), X0, t; full=true)

                       # Find crossing of terminal vertical velocity
                       vs = dot(traj.vs, env.ey)
                       k_0x = findfirst(X[veh.id_v, :]'*env.ey.>=vs)
                       if isnothing(k_0x)
                       msg = string("ERROR: no terminal velocity crossing, ",
                                    "increase time of flight (t_θcst).")
                       error = ArgumentError(msg)
                       throw(error)
                       end
                       t = @k(t, 1, k_0x)
                       t1 = t[end]
                       X = @k(X, 1, k_0x)
                       X[veh.id_τ, :] *= traj.τs/t1

                       # Populate trajectory guess first phase
                       τ2t = (τ) -> τ/τs*t1
                       xc = T_ContinuousTimeTrajectory(t, X, :linear)
                       @k(x, id_phase1) = hcat([
                           sample(xc, τ2t(τ)) for τ in τ_grid[id_phase1]]...)
                       @k(u, id_phase1) = hcat([
                           ctrl(τ2t(τ), pbm) for τ in τ_grid[id_phase1]]...)

                       # >>>>>>>>>>>>>>>><<<<<<<<<<<<<<<
                       # >> Phase 2: terminal descent <<
                       # >>>>>>>>>>>>>>>><<<<<<<<<<<<<<<

                       t2 = 5.0 # Phase 2 duration
                       T_hover = norm(veh.m*env.g)

                       # Straight line interpolate state
                       xs = sample(xc, τ2t(τ_grid[id_phase1][end]))
                       xf = zeros(pbm.nx)
                       xf[veh.id_v] = traj.vf
                       xf[veh.id_m] = xs[veh.id_m]+veh.αe*T_hover*t2
                       xf[veh.id_τ] = 1.0
                       N2 = N-sum(id_phase1)+1
                       X = straightline_interpolate(xs, xf, N2)
                       @k(x, id_phase2) = @k(X, 2, N2)

                       # Constant input
                       u_hov = zeros(pbm.nu)
                       u_hov[veh.id_T] = T_hover
                       U = straightline_interpolate(u_hov, u_hov, N2)
                       @k(u, id_phase2) = @k(U, 2, N2)

                       # >> Parameter guess <<
                       p = T_RealVector(undef, pbm.np)
                       p[veh.id_t1] = t1
                       p[veh.id_t2] = t2

                       return x, u, p
                       end)

    return nothing
end

function _common__set_cost!(pbm::TrajectoryProblem)::Nothing

    problem_set_terminal_cost!(pbm, (x, p, pbm) -> begin
                               veh = pbm.mdl.vehicle
                               traj = pbm.mdl.traj
                               env = pbm.mdl.env
                               m = x[veh.id_m]
                               m_nrml = 10e3
                               return -m/m_nrml
                               end)

    return nothing
end

function _common__set_dynamics!(pbm::TrajectoryProblem)::Nothing

    problem_set_dynamics!(
        pbm,
        # Dynamics f
        (x, u, p, pbm) -> begin
        f = dynamics(x, u, p, pbm)
        return f
        end,
        # Jacobian df/dx
        (x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        env = pbm.mdl.env
        traj = pbm.mdl.traj
        v = x[veh.id_v]
        θ = x[veh.id_θ]
        m = x[veh.id_m]
        τ = x[veh.id_τ]
        T = u[veh.id_T]
        δ = u[veh.id_δ]
        tdil = ((τ<=traj.τs) ? p[veh.id_t1]/traj.τs :
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
        (x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        env = pbm.mdl.env
        traj = pbm.mdl.traj
        v = x[veh.id_v]
        θ = x[veh.id_θ]
        m = x[veh.id_m]
        τ = x[veh.id_τ]
        T = u[veh.id_T]
        δ = u[veh.id_δ]
        tdil = ((τ<=traj.τs) ? p[veh.id_t1]/traj.τs :
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
        (x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        τ = x[veh.id_τ]
        id_t = (τ<=traj.τs) ? veh.id_t1 : veh.id_t2
        F = zeros(pbm.nx, pbm.np)
        F[:, id_t] = pbm.f(x, u, p)/p[id_t]
        F[veh.id_τ, id_t] = 0.0
        return F
        end)

    return nothing
end

function _common__set_convex_constraints!(pbm::TrajectoryProblem)::Nothing

    # Convex path constraints on the state
    problem_set_X!(
        pbm, (τ, x, pbm) -> begin
        traj = pbm.mdl.traj
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        r = x[veh.id_r]
        v = x[veh.id_v]
        C = T_ConvexConeConstraint
        X = [C(dot(v, env.ey), :nonpos)]
        return X
        end)

    # Convex path constraints on the input
    problem_set_U!(
        pbm, (τ, u, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        T = u[veh.id_T]
        δ = u[veh.id_δ]
        flip_phase = τ<=traj.τs
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

function _common__set_nonconvex_constraints!(pbm::TrajectoryProblem)::Nothing

    problem_set_s!(
        pbm,
        # Constraint s
        (x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        env = pbm.mdl.env
        traj = pbm.mdl.traj
        r = x[veh.id_r]
        δd = x[veh.id_δd]
        δ = u[veh.id_δ]
        δdot = u[veh.id_δdot]
        tf = p[veh.id_t1]+p[veh.id_t2]

        s = zeros(7)
        s[1] = tf-traj.tf_max
        s[2] = traj.tf_min-tf
        s[3] = (δ-δd)-δdot*veh.rate_delay
        s[4] = δdot*veh.rate_delay-(δ-δd)
        s[5] = δdot-veh.δdot_max
        s[6] = -veh.δdot_max-δdot
        s[7] = norm(r)*cos(traj.γ_gs)-dot(r, env.ey)
        return s
        end,
        # Jacobian ds/dx
        (x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        env = pbm.mdl.env
        traj = pbm.mdl.traj
        r = x[veh.id_r]

        nrm_r = norm(r)
        ∇nrm_r = (nrm_r<sqrt(eps())) ? zeros(2) : r/nrm_r

        C = zeros(7, pbm.nx)
        C[3, veh.id_δd] = -1.0
        C[4, veh.id_δd] = 1.0
        C[7, veh.id_r] = ∇nrm_r*cos(traj.γ_gs)-env.ey
        return C
        end,
        # Jacobian ds/du
        (x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        D = zeros(7, pbm.nu)
        D[3, veh.id_δ] = 1.0
        D[3, veh.id_δdot] = -veh.rate_delay
        D[4, veh.id_δ] = -1.0
        D[4, veh.id_δdot] = veh.rate_delay
        D[5, veh.id_δdot] = 1.0
        D[6, veh.id_δdot] = -1.0
        return D
        end,
        # Jacobian ds/dp
        (x, u, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        G = zeros(7, pbm.np)
        G[1, veh.id_t1] = 1.0
        G[1, veh.id_t2] = 1.0
        G[2, veh.id_t1] = -1.0
        G[2, veh.id_t2] = -1.0
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
        rhs = zeros(8)
        rhs[1:2] = traj.r0
        rhs[3:4] = traj.v0
        rhs[5] = traj.θ0
        rhs[6] = 0.0
        rhs[7] = 0.0
        rhs[8] = 0.0
        g = x[vcat(veh.id_r,
                   veh.id_v,
                   veh.id_θ,
                   veh.id_ω,
                   veh.id_m,
                   veh.id_τ)]-rhs
        return g
        end,
        # Jacobian dg/dx
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        H = zeros(8, pbm.nx)
        H[1:2, veh.id_r] = I(2)
        H[3:4, veh.id_v] = I(2)
        H[5, veh.id_θ] = 1.0
        H[6, veh.id_ω] = 1.0
        H[7, veh.id_m] = 1.0
        H[8, veh.id_τ] = 1.0
        return H
        end,
        # Jacobian dg/dp
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        K = zeros(8, pbm.np)
        return K
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
        end,
        # Jacobian dg/dp
        (x, p, pbm) -> begin
        veh = pbm.mdl.vehicle
        K = zeros(6, pbm.np)
        return K
        end)

    return nothing
end
