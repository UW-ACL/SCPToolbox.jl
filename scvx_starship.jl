#= Starship landing flip maneuver example using PTR.

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

using ECOS

include("models/starship.jl")
include("core/problem.jl")
include("core/scvx.jl")
include("utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = StarshipProblem()
pbm = TrajectoryProblem(mdl)

# >> Variable dimensions <<
problem_set_dims!(pbm, 8, 3, 1)

# >> Variable scaling <<
# Parameters
problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t,
                      (mdl.traj.tf_min, mdl.traj.tf_max))
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

# >> Initial trajectory guess <<
starship_set_initial_guess!(pbm)

# >> Cost to be minimized <<
problem_set_terminal_cost!(pbm, (x, p, pbm) -> begin
                           veh = pbm.mdl.vehicle
                           m = x[veh.id_m]
                           m_nrml = veh.m
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
                      m = x[veh.id_m]
                      δd = x[veh.id_δd]
                      T = u[veh.id_T]
                      δ = u[veh.id_δ]
                      tdil = p[veh.id_t]

                      ℓeng = -veh.lcg
                      ℓcp = veh.lcp-veh.lcg
                      ei = veh.ei(θ)
                      ej = veh.ej(θ)
                      Tv = T*(-sin(δ)*ei+cos(δ)*ej)
                      MT = ℓeng*T*sin(δ)
                      D = -veh.CD*norm(v)*v
                      MD = -ℓcp*dot(D, ei)

                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = (Tv+D)/veh.m+env.g
                      f[veh.id_θ] = ω
                      f[veh.id_ω] = (MT+MD)/veh.J
                      f[veh.id_m] = veh.αe*T
                      f[veh.id_δd] = (δ-δd)/veh.rate_delay

                      f *= tdil
                      return f
                      end,
                      # Jacobian df/dx
                      (x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      env = pbm.mdl.env
                      v = x[veh.id_v]
                      θ = x[veh.id_θ]
                      m = x[veh.id_m]
                      T = u[veh.id_T]
                      δ = u[veh.id_δ]
                      tdil = p[veh.id_t]

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
                      v = x[veh.id_v]
                      θ = x[veh.id_θ]
                      m = x[veh.id_m]
                      T = u[veh.id_T]
                      δ = u[veh.id_δ]
                      tdil = p[veh.id_t]

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
               δdot = u[veh.id_δdot]
               C = T_ConvexConeConstraint
               U = [C(vcat(veh.δ_max, δ), :l1)]
               return U
               end)

# >> Nonconvex path inequality constraints <<
problem_set_s!(pbm,
               # Constraint s
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               env = pbm.mdl.env
               traj = pbm.mdl.traj
               r = x[veh.id_r]
               δd = x[veh.id_δd]
               T = u[veh.id_T]
               δ = u[veh.id_δ]
               δdot = u[veh.id_δdot]

               h = dot(r, env.ey)
               σ = 1/(1+exp(-traj.kh*(h-traj.h1)))
               T_max = veh.T_max1+σ*(veh.T_max3-veh.T_max1)
               T_min = veh.T_min1+σ*(veh.T_min3-veh.T_min1)

               s = zeros(8)
               s[1] = p[veh.id_t]-traj.tf_max
               s[2] = traj.tf_min-p[veh.id_t]
               s[3] = δ-δd-δdot*veh.rate_delay
               s[4] = δdot*veh.rate_delay-(δ-δd)
               s[5] = δdot-veh.δdot_max
               s[6] = -veh.δdot_max-δdot
               s[7] = T-T_max
               s[8] = T_min-T
               return s
               end,
               # Jacobian ds/dx
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               env = pbm.mdl.env
               traj = pbm.mdl.traj
               r = x[veh.id_r]

               h = dot(r, env.ey)
               σ = 1/(1+exp(-traj.kh*(h-traj.h1)))
               ∇σ = traj.kh*exp(-traj.kh*(h-traj.h1))*σ^2
               ∇h_T_max = ∇σ*(veh.T_max3-veh.T_max1)
               ∇h_T_min = ∇σ*(veh.T_min3-veh.T_min1)

               C = zeros(8, pbm.nx)
               C[3, veh.id_δd] = -1.0
               C[4, veh.id_δd] = 1.0
               C[7, veh.id_r] = -∇h_T_max*env.ey
               C[8, veh.id_r] = ∇h_T_min*env.ey
               return C
               end,
               # Jacobian ds/du
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               D = zeros(8, pbm.nu)
               D[3, veh.id_δ] = 1.0
               D[3, veh.id_δdot] = -veh.rate_delay
               D[4, veh.id_δ] = -1.0
               D[4, veh.id_δdot] = veh.rate_delay
               D[5, veh.id_δdot] = 1.0
               D[6, veh.id_δdot] = -1.0
               D[7, veh.id_T] = 1.0
               D[8, veh.id_T] = -1.0
               return D
               end,
               # Jacobian ds/dp
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               G = zeros(8, pbm.np)
               G[1, veh.id_t] = 1.0
               G[2, veh.id_t] = -1.0
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
                rhs[veh.id_m] = 0.0
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
Nsub = 25
iter_max = 30
λ = 1e2
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
feas_tol = 5e-3
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0, "maxit"=>1000)
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

plot_final_trajectory(mdl, sol)
plot_thrust(mdl, sol)
plot_gimbal(mdl, sol)
plot_convergence(history, "starship")
