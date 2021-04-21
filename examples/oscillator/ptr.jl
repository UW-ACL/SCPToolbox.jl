#= Forced harmonic oscillator with input deadband using PTR.

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

using ECOS

include("common.jl")
include("../../core/ptr.jl")
include("../../models/oscillator.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Common problem definition ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 30

mdl = OscillatorProblem(N)
pbm = TrajectoryProblem(mdl)

define_problem!(pbm, :ptr)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: PTR lgorithm parameters :::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Nsub = 10
iter_max = 10
wvc = 1e2
wtr = 1e-3
ε_abs = -Inf#1e-5
ε_rel = 1e-3/100
feas_tol = 5e-3
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = PTRParameters(N, Nsub, iter_max, wvc, wtr, ε_abs, ε_rel, feas_tol,
                     q_tr, q_exit, solver, solver_options)

# Homotopy parameters
Nhom = 50
hom_κ1 = T_Homotopy(1e-8)
hom_grid = LinRange(0.0, 1.0, Nhom)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Solve trajectory generation problem ::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

ptr_pbm = PTRProblem(pars, pbm)

sol, history = [], []
for i = 1:Nhom
    global sol, history

    mdl.traj.κ1 = hom_κ1(hom_grid[i])
    warm = (i==1) ? nothing : sol[end]

    @printf("[%d/%d] Homotopy (κ=%.2e)\n", i, Nhom, mdl.traj.κ1)

    sol_i, history_i = ptr_solve(ptr_pbm, warm)

    push!(sol, sol_i)
    push!(history, history_i)
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Plot results :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

sol = sol[end]
history = history[end]

plot_timeseries(mdl, sol, history)
plot_deadband(mdl, sol)
plot_convergence(history, "oscillator")
