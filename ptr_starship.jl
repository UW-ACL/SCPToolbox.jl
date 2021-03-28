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
include("core/ptr.jl")
include("utils/helper.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = StarshipProblem()
pbm = TrajectoryProblem(mdl)

# >> Variable dimensions <<
problem_set_dims!(pbm, 9, 3, 1)

# >> Variable scaling <<
# Parameters
problem_advise_scale!(pbm, :parameter, mdl.vehicle.id_t,
                      (mdl.traj.tf_min, mdl.traj.tf_max))
# Inputs
problem_advise_scale!(pbm, :input, mdl.vehicle.id_T,
                      (mdl.vehicle.T_min, mdl.vehicle.T_max3))
problem_advise_scale!(pbm, :input, mdl.vehicle.id_δ,
                      (-mdl.vehicle.δ_max, mdl.vehicle.δ_max))
problem_advise_scale!(pbm, :input, mdl.vehicle.id_φ,
                      (0.0, mdl.vehicle.Afin))
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
                      (0.0, mdl.vehicle.mCH4))
problem_advise_scale!(pbm, :state, mdl.vehicle.id_δd,
                      (-mdl.vehicle.δ_max, mdl.vehicle.δ_max))
problem_advise_scale!(pbm, :state, mdl.vehicle.id_φd,
                      (-mdl.vehicle.α_max, mdl.vehicle.α_max))

# >> Initial trajectory guess <<
starship_set_initial_guess!(pbm)

# >> Cost to be minimized <<
Jw_m = 0.0

problem_set_terminal_cost!(pbm, (x, p, pbm) -> begin
                           veh = pbm.mdl.vehicle
                           m = x[veh.id_m]
                           m_nrml = veh.mCH4
                           return m/m_nrml
                           end)

problem_set_running_cost!(pbm, (x, u, p, pbm) -> begin
                          veh = pbm.mdl.vehicle
                          env = pbm.mdl.env
                          r = x[veh.id_r]
                          θ = x[veh.id_θ]

                          θ_nrml = pi
                          downrange = dot(r, env.ex)
                          downrange_nrml = 100.0

                          # return (θ/pi)^2
                          return (# (downrange/downrange_nrml)^2+
                                  (θ/θ_nrml)^2)
                          end)

# >> Dynamics constraint <<

# Total mass
_m = (m, pbm) -> begin
    veh = pbm.mdl.vehicle
    return veh.mwet#-(1+veh.σ)*m
end

# Center of mass position as a function of fuel mass consumed.
_cg = (m, pbm) -> begin
    veh = pbm.mdl.vehicle
    num = (veh.me*veh.le+veh.ms*veh.ls/2+
           veh.mO2*veh.lO2+veh.mCH4*veh.lCH4)
    den = veh.mwet
    _cg = num/den
    return _cg
end

# Moment of inertia as a function of fuel mass consumed.
_J = (m, lcg, pbm) -> begin
    veh = pbm.mdl.vehicle
    Je = veh.me*(veh.le-lcg)^2
    JCH4 = veh.mCH4*(veh.lCH4-lcg)^2
    JO2 = veh.mO2*(veh.lO2-lcg)^2
    Js = veh.Js0+veh.ms*(veh.ls/2-lcg)^2
    J = Je+JCH4+JO2+Js
    return J
end

# Drag term, returns the value and the gradients
_D = (cd, v, n, ∇n, l, ei, ej, grad) -> begin
    D = -cd*v*dot(v, n)
    MD = -dot(D, ei)*l
    if !grad
        return D, MD
    end
    ∇v_D = -cd*(dot(v, n)*I(2)+v*n')
    ∇θ_D = -cd*v*dot(v, ∇n)
    ∇v_M = -∇v_D'*ei*l
    ∇θ_M = -(dot(∇θ_D, ei)+dot(D, ej))*l

    return D, MD, ∇v_D, ∇θ_D, ∇v_M, ∇θ_M
end

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
                      φd = x[veh.id_φd]
                      T = u[veh.id_T]
                      δ = u[veh.id_δ]
                      φ = u[veh.id_φ]
                      tdil = p[veh.id_t]

                      lcg = _cg(m, pbm)
                      J = _J(m, lcg, pbm)
                      m = _m(m, pbm)
                      dle = veh.le-lcg
                      ei = veh.ei(θ)
                      ej = veh.ej(θ)
                      Tv = T*(-sin(δ)*ei+cos(δ)*ej)
                      Ds0, MDs0 = _D(veh.CDs0, v, -ej, ei, veh.lcp-lcg,
                                     ei, ej, false)
                      Ds1, MDs1 = _D(veh.CDs1, v, -ei, -ej, veh.lcp-lcg,
                                     ei, ej, false)
                      Df, MDf = _D(veh.CDfin*φ, v, -ei, -ej, veh.lf-lcg,
                                   ei, ej, false)

                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = (Tv+Ds0+Ds1+2*Df)/m+env.g
                      f[veh.id_θ] = ω
                      f[veh.id_ω] = (dle*T*sin(δ)+MDs0+MDs1+2*MDf)/J
                      f[veh.id_m] = veh.αe*T
                      f[veh.id_δd] = (δ-δd)/veh.rate_delay
                      f[veh.id_φd] = (φ-φd)/veh.rate_delay

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
                      φ = u[veh.id_φ]
                      tdil = p[veh.id_t]

                      lcg = _cg(m, pbm)
                      J = _J(m, lcg, pbm)
                      m = _m(m, pbm)
                      ei = veh.ei(θ)
                      ej = veh.ej(θ)
                      ∇Tv = T*(-sin(δ)*ej+cos(δ)*-ei)
                      Ds0, MDs0, ∇v_Ds0, ∇θ_Ds0, ∇v_MDs0, ∇θ_MDs0 = _D(
                          veh.CDs0, v, -ej, ei, veh.lcp-lcg, ei, ej, true)
                      Ds1, MDs1, ∇v_Ds1, ∇θ_Ds1, ∇v_MDs1, ∇θ_MDs1 = _D(
                          veh.CDs1, v, -ei, -ej, veh.lcp-lcg, ei, ej, true)
                      Df, MDf, ∇v_Df, ∇θ_Df, ∇v_MDf, ∇θ_MDf = _D(
                          veh.CDfin*φ, v, -ei, -ej, veh.lf-lcg, ei, ej, true)

                      A = zeros(pbm.nx, pbm.nx)
                      A[veh.id_r, veh.id_v] = I(2)
                      A[veh.id_v, veh.id_v] = (∇v_Ds0+∇v_Ds1+2*∇v_Df)/m
                      A[veh.id_v, veh.id_θ] = (∇Tv+∇θ_Ds0+∇θ_Ds1+2*∇θ_Df)/m
                      A[veh.id_θ, veh.id_ω] = 1.0
                      A[veh.id_ω, veh.id_v] = (∇v_MDs0+∇v_MDs1+2*∇v_MDf)/J
                      A[veh.id_ω, veh.id_θ] = (∇θ_MDs0+∇θ_MDs1+2*∇θ_MDf)/J
                      A[veh.id_δd, veh.id_δd] = -1.0/veh.rate_delay
                      A[veh.id_φd, veh.id_φd] = -1.0/veh.rate_delay

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

                      lcg = _cg(m, pbm)
                      J = _J(m, lcg, pbm)
                      m = _m(m, pbm)
                      dle = veh.le-lcg
                      ei = veh.ei(θ)
                      ej = veh.ej(θ)
                      Tu = -sin(δ)*ei+cos(δ)*ej
                      ∇δ_Tv = T*(-cos(δ)*ei-sin(δ)*ej)
                      ∇φ_Df, ∇φ_MDf = _D(veh.CDfin, v, -ei, -ej, veh.lf-lcg,
                                         ei, ej, false)

                      B = zeros(pbm.nx, pbm.nu)
                      B[veh.id_v, veh.id_T] = Tu/m
                      B[veh.id_v, veh.id_δ] = ∇δ_Tv/m
                      B[veh.id_v, veh.id_φ] = (2*∇φ_Df)/m
                      B[veh.id_ω, veh.id_T] = dle/J*sin(δ)
                      B[veh.id_ω, veh.id_δ] = dle/J*T*cos(δ)
                      B[veh.id_ω, veh.id_φ] = (2*∇φ_MDf)/J
                      B[veh.id_m, veh.id_T] = veh.αe
                      B[veh.id_δd, veh.id_δ] = 1.0/veh.rate_delay
                      B[veh.id_φd, veh.id_φ] = 1.0/veh.rate_delay

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
               φ = u[veh.id_φ]
               C = T_ConvexConeConstraint
               U = [C(veh.T_min-T, :nonpos),
                    C(vcat(veh.δ_max, δ), :l1),
                    C(-φ, :nonpos),
                    C(φ-veh.Afin, :nonpos)]
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
               φd = x[veh.id_φd]
               T = u[veh.id_T]
               δ = u[veh.id_δ]
               φ = u[veh.id_φ]

               h = dot(r, env.ey)
               σ = 1/(1+exp(-traj.kh*(h-traj.h1)))
               T_max = veh.T_max1+σ*(veh.T_max3-veh.T_max1)

               s = zeros(7)
               s[1] = δ-δd-veh.β_max*veh.rate_delay
               s[2] = -veh.β_max*veh.rate_delay-δ+δd
               s[3] = φ-φd-veh.α_max*veh.rate_delay
               s[4] = -veh.α_max*veh.rate_delay-φ+φd
               s[5] = p[veh.id_t]-traj.tf_max
               s[6] = traj.tf_min-p[veh.id_t]
               s[7] = T-T_max
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

               C = zeros(7, pbm.nx)
               C[1, veh.id_δd] = -1.0
               C[2, veh.id_δd] = 1.0
               C[3, veh.id_φd] = -1.0
               C[4, veh.id_φd] = 1.0
               C[7, veh.id_r] = -∇h_T_max*env.ey
               return C
               end,
               # Jacobian ds/du
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               D = zeros(7, pbm.nu)
               D[1, veh.id_δ] = 1.0
               D[2, veh.id_δ] = -1.0
               D[3, veh.id_φ] = 1.0
               D[4, veh.id_φ] = -1.0
               D[7, veh.id_T] = 1.0
               return D
               end,
               # Jacobian ds/dp
               (x, u, p, pbm) -> begin
               veh = pbm.mdl.vehicle
               G = zeros(7, pbm.np)
               G[5, veh.id_t] = 1.0
               G[6, veh.id_t] = -1.0
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
iter_max = 10
wvc = 1e5
wtr = 0.1
ε_abs = 1e-5
ε_rel = 0.01/100
feas_tol = 5e-3
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0, "maxit"=>1000)
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
plot_fin(mdl, sol)
plot_convergence(history, "starship")
