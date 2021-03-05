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

using LinearAlgebra
using ECOS
using Plots

include("models/quadrotor.jl")
include("core/problem.jl")
include("core/scvx.jl")
include("utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = QuadrotorProblem()
pbm = TrajectoryProblem(mdl)

# Variable dimensions
problem_set_dims!(pbm, 7, 4, 1)

# Initial trajectory guess
problem_set_guess!(pbm, (N, pbm) -> begin
                   return quadrotor_initial_guess(N, pbm)
                   end)

# Cost to be minimized
problem_set_cost!(pbm;
                  # Running cost
                  Γ = (x, u, p, pbm) -> begin
                  σ = u[pbm.mdl.vehicle.id_σ]
                  return σ^2
                  end)

# Dynamics constraint
problem_set_dynamics!(pbm,
                      # Dynamics f
                      (x, u, p, pbm) -> begin
                      g = pbm.mdl.env.g
                      veh = pbm.mdl.vehicle
                      v = x[veh.id_v]
                      uu = u[veh.id_u]
                      tdil = p[veh.id_pt] # Time dilation
                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = uu+g
                      f *= tdil
                      return f
                      end,
                      # Jacobian df/dx
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      A = zeros(pbm.nx, pbm.nx)
                      A[veh.id_r, veh.id_v] = I(3)
                      A *= tdil
                      return A
                      end,
                      # Jacobian df/du
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      B = zeros(pbm.nx, pbm.nu)
                      B[veh.id_v, veh.id_u] = I(3)
                      B *= tdil
                      return B
                      end,
                      # Jacobian df/dp
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      F = zeros(pbm.nx, pbm.np)
                      F[:, veh.id_pt] = pbm.f(x, u, p)/tdil
                      return F
                      end)

# Convex path constraints on the state
problem_set_X!(pbm, (x, pbm) -> begin
               traj = pbm.mdl.traj
               veh = pbm.mdl.vehicle
               C = T_ConvexConeConstraint
               X = [C(x[veh.id_xt]-traj.tf_max, :nonpositiveorthant),
                    C(traj.tf_min-x[veh.id_xt], :nonpositiveorthant)]
               return X
               end)

# Convex path constraints on the input
problem_set_U!(pbm, (u, pbm) -> begin
               veh = pbm.mdl.vehicle
               uu = u[veh.id_u]
               σ = u[veh.id_σ]
               C = T_ConvexConeConstraint
               U = [C(veh.u_min-σ, :nonpositiveorthant),
                    C(σ-veh.u_max, :nonpositiveorthant),
                    C(vcat(σ, uu), :secondordercone),
                    C(σ*cos(veh.tilt_max)-uu[3], :nonpositiveorthant)]
               return U
               end)

# Nonconvex path inequality constraints
problem_set_s!(pbm,
               # Constraint s
               (x, u, p, pbm) -> begin
               env = pbm.mdl.env
               veh = pbm.mdl.vehicle
               s = zeros(env.n_obs)
               for i = 1:env.n_obs
               # ---
               E = env.obs[i]
               r = x[veh.id_r]
               s[i] = 1-E(r)
               # ---
               end
               return s
               end,
               # Jacobian ds/dx
               (x, u, p, pbm) -> begin
               env = pbm.mdl.env
               veh = pbm.mdl.vehicle
               C = zeros(env.n_obs, pbm.nx)
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
               D = zeros(env.n_obs, pbm.nu)
               return D
               end,
               # Jacobian ds/dp
               (x, u, p, pbm) -> begin
               env = pbm.mdl.env
               G = zeros(env.n_obs, pbm.np)
               return G
               end)

# Initial boundary conditions
problem_set_bc!(pbm, :ic,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                tdil = p[veh.id_pt]
                rhs = zeros(pbm.nx)
                rhs[veh.id_r] = traj.r0
                rhs[veh.id_v] = traj.v0
                rhs[veh.id_xt] = tdil
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
                K[veh.id_xt, veh.id_pt] = -1.0
                return K
                end)

# Terminal boundary conditions
problem_set_bc!(pbm, :tc,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                tdil = p[veh.id_pt]
                rhs = zeros(pbm.nx)
                rhs[veh.id_r] = traj.rf
                rhs[veh.id_v] = traj.vf
                rhs[veh.id_xt] = tdil
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
                K[veh.id_xt, veh.id_pt] = -1.0
                return K
                end)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: SCvx algorithm parameters ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30
Nsub = 15
iter_max = 20
λ = 1e3
ρ_0 = 0.0
ρ_1 = 0.1
ρ_2 = 0.7
β_sh = 2.0
β_gr = 2.0
η_init = 1.0
η_lb = 1e-3
η_ub = 10.0
ε_abs = 0.0#1e-4
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

pyplot()
plot_trajectory_history(mdl, history)
plot_final_trajectory(mdl, sol)
plot_input_norm(mdl, sol)
plot_tilt_angle(mdl, sol)
plot_convergence(mdl, history)
