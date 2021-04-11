#= 6-Degree of Freedom free-flyer example using SCvx.

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
include("../../utils/helper.jl")
include("../../core/problem.jl")
include("../../core/scvx.jl")
include("../../models/freeflyer.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = FreeFlyerProblem()
pbm = TrajectoryProblem(mdl)

define_problem!(pbm)

# >> Special numerical integration <<

# Quaternion re-normalization on numerical integration step
problem_set_integration_action!(
    pbm, mdl.vehicle.id_q,
    (x, pbm) -> begin
    xn = x/norm(x)
    return xn
    end)

# >> Running cost to be minimized <<
problem_set_running_cost!(pbm, (x, u, p, pbm) -> begin
                          traj = pbm.mdl.traj
                          veh = pbm.mdl.vehicle
                          T_max_sq = veh.T_max^2
                          M_max_sq = veh.M_max^2
                          T = u[veh.id_T]
                          M = u[veh.id_M]
                          γ = traj.γ
                          return (1-γ)*((T'*T)/T_max_sq+(M'*M)/M_max_sq)
                          end)

# >> Dynamics constraint <<
problem_set_dynamics!(pbm,
                      # Dynamics f
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_t] # Time dilation
                      v = x[veh.id_v]
                      q = T_Quaternion(x[veh.id_q])
                      ω = x[veh.id_ω]
                      T = u[veh.id_T]
                      M = u[veh.id_M]
                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = T/veh.m
                      f[veh.id_q] = 0.5*vec(q*ω)
                      f[veh.id_ω] = veh.J\(M-cross(ω, veh.J*ω))
                      f *= tdil
                      return f
                      end,
                      # Jacobian df/dx
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_t]
                      v = x[veh.id_v]
	              q = T_Quaternion(x[veh.id_q])
	              ω = x[veh.id_ω]
                      dfqdq = 0.5*skew(T_Quaternion(ω), :R)
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
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_t]
                      B = zeros(pbm.nx, pbm.nu)
                      B[veh.id_v, veh.id_T] = (1.0/veh.m)*I(3)
                      B[veh.id_ω, veh.id_M] = veh.J\I(3)
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
    r = x[veh.id_r]
    s = zeros(env.n_obs+3)
    # Ellipsoidal obstacles
    for i = 1:env.n_obs
    # ---
    E = env.obs[i]
    s[i] = 1-E(r)
    # ---
    end
    # Space station flight space
    d_iss, _ = signed_distance(env.iss, r; t=traj.hom,
                               a=traj.sdf_pwr)
    s[end-2] = -d_iss
    # Flight time
    s[end-1] = p[veh.id_t]-traj.tf_max
    s[end] = traj.tf_min-p[veh.id_t]
    return s
    end,
    # Jacobian ds/dx
    (x, u, p, pbm) -> begin
    env = pbm.mdl.env
    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj
    r = x[veh.id_r]
    C = zeros(env.n_obs+3, pbm.nx)
    # Ellipsoidal obstacles
    for i = 1:env.n_obs
    # ---
    E = env.obs[i]
    C[i, veh.id_r] = -∇(E, r)
    # ---
    end
    # Space station flight space
    _, ∇d_iss = signed_distance(env.iss, r; t=traj.hom,
                                a=traj.sdf_pwr)
    C[end-2, veh.id_r] = -∇d_iss
    return C
    end,
    # Jacobian ds/du
    (x, u, p, pbm) -> begin
    env = pbm.mdl.env
    D = zeros(env.n_obs+3, pbm.nu)
    return D
    end,
    # Jacobian ds/dp
    (x, u, p, pbm) -> begin
    veh = pbm.mdl.vehicle
    env = pbm.mdl.env
    G = zeros(env.n_obs+3, pbm.np)
    G[end-1, veh.id_t] = 1.0
    G[end, veh.id_t] = -1.0
    return G
    end)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: SCvx algorithm parameters ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 50
Nsub = 15
iter_max = 50
λ = 1e3
ρ_0 = 0.0
ρ_1 = 0.1
ρ_2 = 0.7
β_sh = 2.0
β_gr = 2.0
η_init = 1.0
η_lb = 1e-3
η_ub = 10.0
ε_abs = 1e-6
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
plot_timeseries(mdl, sol)
plot_obstacle_constraints(mdl, sol)
plot_convergence(history, "freeflyer")
