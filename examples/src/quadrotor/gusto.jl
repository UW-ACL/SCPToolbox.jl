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

include("common.jl")
include("../../core/gusto.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Trajectory optimization problem ::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = QuadrotorProblem()
pbm = TrajectoryProblem(mdl)

define_problem!(pbm, :gusto)

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
