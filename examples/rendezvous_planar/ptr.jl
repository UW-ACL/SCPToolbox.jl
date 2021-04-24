#= Planar spacecraft rendezvous using PTR.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

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
include("../../core/ptr.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Common problem definition ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = PlanarRendezvousProblem()
pbm = TrajectoryProblem(mdl)

define_problem!(pbm, :ptr)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: PTR lgorithm parameters :::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30
Nsub = 10
iter_max = 20
wvc = 5e2
# wtr = 1e-4
wtr = 5e-2
ε_abs = 1e-5
ε_rel = 1e-3/100
feas_tol = 5e-3
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = PTRParameters(N, Nsub, iter_max, wvc, wtr, ε_abs, ε_rel, feas_tol,
                     q_tr, q_exit, solver, solver_options)

# Homotopy parameters
Nhom = 10
hom_κ1 = T_Homotopy(1e-8)
hom_grid = LinRange(0.0, 1.0, Nhom)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Solve trajectory generation problem ::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

ptr_pbm = PTRProblem(pars, pbm)

# TODO homotopy
sol, history = ptr_solve(ptr_pbm)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Plot results :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

plot_final_trajectory(mdl, sol)
plot_attitude(mdl, sol)
plot_thrusts(mdl, sol)
plot_convergence(history, "rendezvous_planar")
