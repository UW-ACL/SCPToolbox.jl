#= 6-Degree of Freedom free-flyer example using GuSTO.

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

include("models/freeflyer.jl")
include("core/problem.jl")
include("core/gusto.jl")
include("utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = FreeFlyerProblem()
pbm = TrajectoryProblem(mdl)

# >> Variable dimensions <<
problem_set_dims!(pbm, 13, 6, 1)

# >> Variable scaling <<
veh, traj = mdl.vehicle, mdl.traj
for i in veh.id_r
    min_pos = min(traj.r0[i], traj.rf[i])
    max_pos = max(traj.r0[i], traj.rf[i])
    problem_advise_scale!(pbm, :state, i, (min_pos, max_pos))
end
problem_advise_scale!(pbm, :parameter, veh.id_pt, (traj.tf_min, traj.tf_max))

# >> Special numerical integration <<

# Quaternion re-normalization on numerical integration step
problem_set_integration_action!(pbm, veh.id_q, (x, pbm) -> begin
                                xn = x/norm(x)
                                return xn
                                end)

# >> Initial trajectory guess <<
freeflyer_set_initial_guess!(pbm)

# >> Cost to be minimized <<
problem_set_terminal_cost!(pbm, (x, p, pbm) -> begin
                           traj = pbm.mdl.traj
                           veh = pbm.mdl.vehicle
                           return traj.γ*p[veh.id_pt]/traj.tf_max
                           end)

problem_set_running_cost!(pbm,
                          # Input quadratic penalty S
                          (p, pbm) -> begin
                          traj = pbm.mdl.traj
                          veh = pbm.mdl.vehicle
                          T_max_sq = veh.T_max^2
                          M_max_sq = veh.M_max^2
                          γ = traj.γ
                          S = zeros(pbm.nu, pbm.nu)
                          S[veh.id_T, veh.id_T] = (1-γ)*1/T_max_sq
                          S[veh.id_M, veh.id_M] = (1-γ)*1/M_max_sq
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
_gusto_freeflyer__f = (x, p, pbm) -> begin
    # TODO
    # return f
end

problem_set_dynamics!(pbm,
                      # Dynamics f
                      (x, p, pbm) -> begin
                      return _gusto_freeflyer__f(x, p, pbm)
                      end,
                      # Jacobian df/dx
                      (x, p, pbm) -> begin
                      # TODO
                      # return A
                      end,
                      # Jacobian df/dp
                      (x, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_t]
                      F = [zeros(pbm.nx, pbm.np) for i=1:pbm.nu+1]
                      _f = _gusto_freeflyer__f(x, p, pbm)
                      for i = 1:pbm.nu+1
                      # ---
                      F[i][:, veh.id_t] = _f[i]/tdil
                      # ---
                      end
                      return F
                      end)

# >> Convex path constraints on the state <<
problem_set_X!(pbm, (x, pbm) -> begin
               traj = pbm.mdl.traj
               veh = pbm.mdl.vehicle
               C = T_ConvexConeConstraint
               X = [C(vcat(veh.v_max, x[veh.id_v]), :soc),
                    C(vcat(veh.ω_max, x[veh.id_ω]), :soc)]
               return X
               end)

# >> Convex path constraints on the input <<
problem_set_U!(pbm, (u, pbm) -> begin
               veh = pbm.mdl.vehicle
               C = T_ConvexConeConstraint
               U = [C(vcat(veh.T_max, u[veh.id_T]), :soc),
                    C(vcat(veh.M_max, u[veh.id_M]), :soc)]
               return U
               end)

# >> Nonconvex path inequality constraints <<
problem_set_s!(pbm,
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
               s[end-2] = d_iss
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
               C[end-2, veh.id_r] = ∇d_iss
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

# >> Initial boundary conditions <<
problem_set_bc!(pbm, :ic,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                tdil = p[veh.id_pt]
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

# >> Terminal boundary conditions <<
problem_set_bc!(pbm, :tc,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                tdil = p[veh.id_pt]
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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: GuSTO algorithm parameters :::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30
Nsub = 15
iter_max = 50
ω = 500.0
λ_init = 13e3
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
iter_μ = 5
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

# TODO
