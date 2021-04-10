#= Quadrotor obstacle avoidance example using GuSTO.

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
include("../../core/gusto.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = QuadrotorProblem()
pbm = TrajectoryProblem(mdl)

define_problem!(pbm)

# >> Running cost to be minimized <<
problem_set_running_cost!(
    pbm,
    # Input quadratic penalty S
    (p, pbm) -> begin
    veh = pbm.mdl.vehicle
    env = pbm.mdl.env
    traj = pbm.mdl.traj
    hover = norm(env.g)
    γ = traj.γ
    S = zeros(pbm.nu, pbm.nu)
    S[veh.id_σ, veh.id_σ] = (1-γ)*1/hover^2
    return S
    end,
    # Jacobian dS/dp
    nothing,
    # Input-affine penalty ℓ
    nothing,
    # Jacobian dℓ/dx
    nothing,
    # Jacobian dℓ/dp
    nothing,
    # Additive penalty g
    nothing,
    # Jacobian dg/dx
    nothing,
    # Jacobian dg/dp
    nothing)

# >> Dynamics constraint <<

# The input-affine dynamics function
_gusto_quadrotor__f = (x, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    g = pbm.mdl.env.g
    v = x[veh.id_v]
    tdil = p[veh.id_t]
    f = [zeros(pbm.nx) for i=1:pbm.nu+1]
    f[1][veh.id_r] = v
    f[1][veh.id_v] = g
    for j = 1:length(veh.id_u)
        # ---
        i = veh.id_u[j]
        f[i+1][veh.id_v[j]] = 1.0
        # ---
    end
    f = [_f*tdil for _f in f]
    return f
end

problem_set_dynamics!(
    pbm,
    # Dynamics f
    (x, p, pbm) -> begin
    return _gusto_quadrotor__f(x, p, pbm)
    end,
    # Jacobian df/dx
    (x, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    tdil = p[veh.id_t]
    A = [zeros(pbm.nx, pbm.nx) for i=1:pbm.nu+1]
    A[1][veh.id_r, veh.id_v] = I(3)
    A = [_A*tdil for _A in A]
    return A
    end,
    # Jacobian df/dp
    (x, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    tdil = p[veh.id_t]
    F = [zeros(pbm.nx, pbm.np) for i=1:pbm.nu+1]
    _f = _gusto_quadrotor__f(x, p, pbm)
    for i = 1:pbm.nu+1
    # ---
    F[i][:, veh.id_t] = _f[i]/tdil
    # ---
    end
    return F
    end)

# >> Nonconvex path inequality constraints <<
problem_set_s!(
    pbm,
    # Constraint s
    (x, p, pbm) -> begin
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
    (x, p, pbm) -> begin
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
    # Jacobian ds/dp
    (x, p, pbm) -> begin
    env = pbm.mdl.env
    veh = pbm.mdl.vehicle
    G = zeros(env.n_obs+2, pbm.np)
    G[end-1, veh.id_t] = 1.0
    G[end, veh.id_t] = -1.0
    return G
    end)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: GuSTO algorithm parameters :::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30
Nsub = 15
iter_max = 50
ω = 500.0
λ_init = 1e4
λ_max = 1e9
ρ_0 = 0.1
ρ_1 = 0.5
β_sh = 2.0
β_gr = 2.0
γ_fail = 5.0
η_init = 1.0
η_lb = 1e-3
η_ub = 10.0
μ = 0.8
iter_μ = 6
ε_abs = 1e-5
ε_rel = 0.01/100
feas_tol = 1e-3
pen = :quad
hom = 100.0
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = GuSTOParameters(N, Nsub, iter_max, ω, λ_init, λ_max, ρ_0, ρ_1, β_sh,
                       β_gr, γ_fail, η_init, η_lb, η_ub, μ, iter_μ, ε_abs,
                       ε_rel, feas_tol, pen, hom, q_tr, q_exit, solver,
                       solver_options)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Solve trajectory generation problem ::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

gusto_pbm = GuSTOProblem(pars, pbm)
sol, history = gusto_solve(gusto_pbm)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Plot results :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

plot_trajectory_history(mdl, history)
plot_final_trajectory(mdl, sol)
plot_input_norm(mdl, sol)
plot_tilt_angle(mdl, sol)
plot_convergence(history, "quadrotor")
