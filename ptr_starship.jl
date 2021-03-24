#= Starship landing flip maneuver example using PTR.

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

include("models/starship.jl")
include("core/problem.jl")
include("core/ptr.jl")
include("utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = StarshipProblem()
pbm = TrajectoryProblem(mdl)

# >> Variable dimensions <<
problem_set_dims!(pbm, 8, 2, 1)

# >> Variable scaling <<
# Parameters
problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t,
                      (mdl.traj.tf_min, mdl.traj.tf_max))
# Inputs
problem_advise_scale!(pbm, :input, mdl.vehicle.id_T,
                      (mdl.vehicle.T_min, mdl.vehicle.T_max))
problem_advise_scale!(pbm, :input, mdl.vehicle.id_δ,
                      (-mdl.vehicle.δ_max, mdl.vehicle.δ_max))
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
mw = mdl.vehicle.m_wet
mf = mw+10*mdl.vehicle.αe*mdl.vehicle.T_max
problem_advise_scale!(pbm, :state, mdl.vehicle.id_m, (mf, mw))
problem_advise_scale!(pbm, :state, mdl.vehicle.id_γ,
                      (-mdl.vehicle.δ_max, mdl.vehicle.δ_max))

# >> Initial trajectory guess <<
starship_set_initial_guess!(pbm)

# >> Cost to be minimized <<
problem_set_terminal_cost!(pbm, (x, p, pbm) -> begin
                           veh = pbm.mdl.vehicle
                           m = x[veh.id_m]
                           m_nrml = veh.m_wet
                           return -m/m_nrml
                           end)

# >> Dynamics constraint <<
problem_set_dynamics!(pbm,
                      # Dynamics f
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      env = pbm.mdl.env
                      v = x[veh.id_v]
                      θ = x[veh.id_θ]
                      ω = x[veh.id_ω]
                      γ = x[veh.id_γ]
                      T = u[veh.id_T]
                      δ = u[veh.id_δ]
                      tdil = p[veh.id_t]
                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = T/veh.m_wet*(cos(θ+δ)*veh.ej-
                                                 sin(θ+δ)*veh.ei)+env.g
                      f[veh.id_θ] = ω
                      f[veh.id_ω] = -veh.lg/veh.J*T*sin(δ)
                      f[veh.id_m] = veh.αe*T
                      f[veh.id_γ] = (δ-γ)/veh.δ_delay
                      f *= tdil
                      return f
                      end,
                      # Jacobian df/dx
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      θ = x[veh.id_θ]
                      T = u[veh.id_T]
                      δ = u[veh.id_δ]
                      tdil = p[veh.id_t]
                      A = zeros(pbm.nx, pbm.nx)
                      A[veh.id_r, veh.id_v] = I(2)
                      A[veh.id_v, veh.id_θ] = -T/veh.m_wet*(sin(θ+δ)*veh.ej+
                                                            cos(θ+δ)*veh.ei)
                      A[veh.id_θ, veh.id_ω] = 1.0
                      A[veh.id_γ, veh.id_γ] = -1.0/veh.δ_delay
                      A *= tdil
                      return A
                      end,
                      # Jacobian df/du
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      θ = x[veh.id_θ]
                      T = u[veh.id_T]
                      δ = u[veh.id_δ]
                      tdil = p[veh.id_t]
                      B = zeros(pbm.nx, pbm.nu)
                      B[veh.id_v, veh.id_T] = (cos(θ+δ)*veh.ej-
                                               sin(θ+δ)*veh.ei)/veh.m_wet
                      B[veh.id_v, veh.id_δ] = -T/veh.m_wet*(sin(θ+δ)*veh.ej+
                                                            cos(θ+δ)*veh.ei)
                      B[veh.id_ω, veh.id_T] = -veh.lg/veh.J*sin(δ)
                      B[veh.id_ω, veh.id_δ] = -veh.lg/veh.J*T*cos(δ)
                      B[veh.id_m, veh.id_T] = veh.αe
                      B[veh.id_γ, veh.id_δ] = 1.0/veh.δ_delay
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

# >> Convex path constraints on the state <<
problem_set_X!(pbm, (x, pbm) -> begin
               traj = pbm.mdl.traj
               env = pbm.mdl.env
               veh = pbm.mdl.vehicle
               r = x[veh.id_r]
               C = T_ConvexConeConstraint
               X = [C(vcat(dot(r, env.ey)/cos(traj.γ_gs), r), :soc)]
               return X
               end)

# >> Convex path constraints on the input <<
problem_set_U!(pbm, (u, pbm) -> begin
               veh = pbm.mdl.vehicle
               T = u[veh.id_T]
               δ = u[veh.id_δ]
               C = T_ConvexConeConstraint
               U = [C(veh.T_min-T, :nonpos),
                    C(T-veh.T_max, :nonpos),
                    C(vcat(veh.δ_max, δ), :l1)]
               return U
               end)

# >> Nonconvex path inequality constraints <<
problem_set_s!(pbm,
               # Constraint s
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               traj = pbm.mdl.traj
               γ = x[veh.id_γ]
               δ = u[veh.id_δ]
               s = zeros(4)
               s[1] = δ-γ-veh.β_max*veh.δ_delay
               s[2] = -veh.β_max*veh.δ_delay-δ+γ
               s[3] = p[veh.id_t]-traj.tf_max
               s[4] = traj.tf_min-p[veh.id_t]
               return s
               end,
               # Jacobian ds/dx
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               C = zeros(4, pbm.nx)
               C[1, veh.id_γ] = -1.0
               C[2, veh.id_γ] = 1.0
               return C
               end,
               # Jacobian ds/du
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               D = zeros(4, pbm.nu)
               D[1, veh.id_δ] = 1.0
               D[2, veh.id_δ] = -1.0
               return D
               end,
               # Jacobian ds/dp
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               G = zeros(4, pbm.np)
               G[3, veh.id_t] = 1.0
               G[4, veh.id_t] = -1.0
               return G
               end)

# >> Initial boundary conditions <<
problem_set_bc!(pbm, :ic,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                rhs = zeros(7)
                rhs[veh.id_r] = traj.r0
                rhs[veh.id_v] = traj.v0
                rhs[veh.id_θ] = traj.θ0
                rhs[veh.id_ω] = 0.0
                rhs[veh.id_m] = veh.m_wet
                g = x[1:7]-rhs
                return g
                end,
                # Jacobian dg/dx
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                H = zeros(7, pbm.nx)
                H[veh.id_r, veh.id_r] = I(2)
                H[veh.id_v, veh.id_v] = I(2)
                H[veh.id_θ, veh.id_θ] = 1.0
                H[veh.id_ω, veh.id_ω] = 1.0
                H[veh.id_m, veh.id_m] = 1.0
                return H
                end,
                # Jacobian dg/dp
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                K = zeros(7, pbm.np)
                return K
                end)

# >> Terminal boundary conditions <<
problem_set_bc!(pbm, :tc,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                rhs = zeros(6)
                rhs[veh.id_r] = zeros(2)
                rhs[veh.id_v] = traj.vf
                rhs[veh.id_θ] = 0.0
                rhs[veh.id_ω] = 0.0
                g = x[1:6]-rhs
                return g
                end,
                # Jacobian dg/dx
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                H = zeros(6, pbm.nx)
                H[veh.id_r, veh.id_r] = I(2)
                H[veh.id_v, veh.id_v] = I(2)
                H[veh.id_θ, veh.id_θ] = 1.0
                H[veh.id_ω, veh.id_ω] = 1.0
                return H
                end,
                # Jacobian dg/dp
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                K = zeros(6, pbm.np)
                return K
                end)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: SCvx algorithm parameters ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30
Nsub = 15
iter_max = 50
wvc = 1e5
wtr = 1e1
ε_abs = 1e-5
ε_rel = 0.01/100
feas_tol = 1e-3
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = PTRParameters(N, Nsub, iter_max, wvc, wtr, ε_abs, ε_rel, feas_tol,
                     q_tr, q_exit, solver, solver_options)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Solve trajectory generation problem ::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

ptr_pbm = PTRProblem(pars, pbm)
sol, history = ptr_solve(ptr_pbm)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Plot results :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

plot_final_trajectory(mdl, sol)
plot_thrust(mdl, sol)
plot_gimbal(mdl, sol)
