#= GuSTO algorithm data structures and methods using GuSTO.

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

include("models/quadrotor.jl")
include("core/problem.jl")
include("core/gusto.jl")
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
                  # Input quadratic penalty, S
                  S = (p, pbm) -> begin
                  veh = pbm.mdl.vehicle
                  S = zeros(pbm.nu, pbm.nu)
                  S[veh.id_σ, veh.id_σ] = p[veh.id_pt]
                  return S
                  end,
                  # Jacobian dS/dp
                  ∇pS = (p, pbm) -> begin
                  veh = pbm.mdl.vehicle
                  ∇pS = [zeros(pbm.nu, pbm.nu) for i=1:pbm.np]
                  ∇pS[veh.id_pt][veh.id_σ, veh.id_σ] = 1.0
                  return ∇pS
                  end)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: GuSTO algorithm parameters :::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30
Nsub = 15
iter_max = 20
ω = 1e3
λ_init = 13e3
λ_max = 1e9
ρ_0 = 5.0
ρ_1 = 20.0
β_sh = 2.0
β_gr = 2.0
γ_fail = 5.0
η_init = 1.0
η_lb = 1e-3
η_ub = 10.0
ε = 1e-3
feas_tol = 1e-3
pen = :quad
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = GuSTOParameters(N, Nsub, iter_max, ω, λ_init, λ_max, ρ_0, ρ_1, β_sh,
                       β_gr, γ_fail, η_init, η_lb, η_ub, ε, feas_tol, pen,
                       q_tr, q_exit, solver, solver_options)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Solve trajectory generation problem ::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

gusto_pbm = GuSTOProblem(pars, pbm)
sol, history = gusto_solve(gusto_pbm)
