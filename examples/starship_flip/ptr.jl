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

include("common.jl")
include("../../core/ptr.jl")
include("../../models/starship.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Common problem definition ::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

mdl = StarshipProblem()
pbm = TrajectoryProblem(mdl)

define_problem!(pbm)

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: PTR algorithm parameters :::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

N = 31
Nsub = 100
iter_max = 15
wvc = 1e3
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

plot_trajectory_history(mdl, history)
plot_final_trajectory(mdl, sol)
plot_velocity(mdl, sol)
plot_thrust(mdl, sol)
plot_gimbal(mdl, sol)
plot_convergence(history, "starship")
