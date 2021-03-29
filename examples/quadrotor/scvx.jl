#= Quadrotor obstacle avoidance example using SCvx.

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

using ECOS

include("common.jl")
include("../../models/quadrotor.jl")
include("../../core/problem.jl")
include("../../core/scvx.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = QuadrotorProblem()
pbm = TrajectoryProblem(mdl)

define_problem!(pbm)

# >> Cost to be minimized <<
problem_set_terminal_cost!(
    pbm, (x, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj
    tdil = p[veh.id_t]
    tdil_max = traj.tf_max
    γ = traj.γ
    return γ*(tdil/tdil_max)^2
    end)

problem_set_running_cost!(
    pbm, (x, u, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    env = pbm.mdl.env
    traj = pbm.mdl.traj
    σ = u[veh.id_σ]
    hover = norm(env.g)
    γ = traj.γ
    return (1-γ)*(σ/hover)^2
    end)

# >> Dynamics constraint <<
problem_set_dynamics!(
    pbm,
    # Dynamics f
    (x, u, p, pbm) -> begin
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
    (x, u, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    tdil = p[veh.id_t]
    A = zeros(pbm.nx, pbm.nx)
    A[veh.id_r, veh.id_v] = I(3)
    A *= tdil
    return A
    end,
    # Jacobian df/du
    (x, u, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    tdil = p[veh.id_t]
    B = zeros(pbm.nx, pbm.nu)
    B[veh.id_v, veh.id_u] = I(3)
    B *= tdil
    return B
    end,
    # Jacobian df/dp
    (x, u, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    tdil = p[veh.id_t]
    F = zeros(pbm.nx, pbm.np)
    F[:, veh.id_t] = pbm.f(x, u, p)/tdil
    return F
    end)

# >> Nonconvex path inequality constraints <<
problem_set_s!(
        pbm,
        # Constraint s
        (x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        s = zeros(env.n_obs+2)
        for i = 1:env.n_obs
        # ---
        E = env.obs[i]
        r = x[veh.id_r]
        s[i] = 1-E(r)
        # ---
        end
        s[end-1] = p[veh.id_t]-traj.tf_max
        s[end] = traj.tf_min-p[veh.id_t]
        return s
        end,
        # Jacobian ds/dx
        (x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        C = zeros(env.n_obs+2, pbm.nx)
        for i = 1:env.n_obs
        # ---
        E = env.obs[i]
        r = x[veh.id_r]
        C[i, veh.id_r] = -∇(E, r)
        # ---
        end
        return C
        end,
        # Jacobian ds/du
        (x, u, p, pbm) -> begin
        env = pbm.mdl.env
        D = zeros(env.n_obs+2, pbm.nu)
        return D
        end,
        # Jacobian ds/dp
        (x, u, p, pbm) -> begin
        env = pbm.mdl.env
        veh = pbm.mdl.vehicle
        G = zeros(env.n_obs+2, pbm.np)
        G[end-1, veh.id_t] = 1.0
        G[end, veh.id_t] = -1.0
        return G
        end)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: SCvx algorithm parameters ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30
Nsub = 15
iter_max = 50
λ = 30.0
ρ_0 = 0.0
ρ_1 = 0.1
ρ_2 = 0.7
β_sh = 2.0
β_gr = 2.0
η_init = 1.0
η_lb = 1e-3
η_ub = 10.0
ε_abs = 1e-5
ε_rel = 0.01/100
feas_tol = 1e-3
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = SCvxParameters(N, Nsub, iter_max, λ, ρ_0, ρ_1, ρ_2, β_sh, β_gr,
                      η_init, η_lb, η_ub, ε_abs, ε_rel, feas_tol, q_tr,
                      q_exit, solver, solver_options)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Solve trajectory generation problem ::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

scvx_pbm = SCvxProblem(pars, pbm)
sol, history = scvx_solve(scvx_pbm)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Plot results :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

plot_trajectory_history(mdl, history)
plot_final_trajectory(mdl, sol)
plot_input_norm(mdl, sol)
plot_tilt_angle(mdl, sol)
plot_convergence(history, "quadrotor")
