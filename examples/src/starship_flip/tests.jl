#= Tests for starship flip.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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
using Printf
using Test
using Parser
using Utils
import Solvers

export ptr

const PTR = Solvers.PTR

function ptr()::Nothing

    # Problem definition
    mdl = StarshipProblem()
    pbm = TrajectoryProblem(mdl)
    define_problem!(pbm, :ptr)

    # PTR algorithm parameters
    N = 31
    Nsub = 100
    iter_max = 15
    disc_method = FOH
    wvc = 1e3
    wtr = 0.1
    ε_abs = 1e-5
    ε_rel = 0.01/100
    feas_tol = 5e-3
    q_tr = Inf
    q_exit = Inf
    solver = ECOS
    solver_options = Dict("verbose"=>0, "maxit"=>1000)
    pars = PTR.Parameters(N, Nsub, iter_max, disc_method, wvc, wtr, ε_abs,
                          ε_rel, feas_tol, q_tr, q_exit, solver,
                          solver_options)

    test_single(mdl, pbm, pars)

    return nothing
end

"""
    test_single(pbm, pars)

Compute a single trajectory.

# Arguments
- `mdl`: the starship parameters.
- `pbm`: the trajectory problem definition.
- `pars`: the algorithm parameters.

# Returns
- `sol`: the trajectory solution.
- `history`: the iterate history.
"""
function test_single(mdl::StarshipProblem,
                     pbm::TrajectoryProblem,
                     pars::PTR.Parameters)::Tuple{SCPSolution,
                                                  SCPHistory}

    test_heading("Single trajectory")

    # Create problem
    ptr = PTR.create(pars, pbm)

    # Solve problem
    sol, history = PTR.solve(ptr)

    @assert sol.status == @sprintf("%s", SCP_SOLVED)

    # Make plots
    plot_trajectory_history(mdl, history)
    plot_final_trajectory(mdl, sol)
    plot_velocity(mdl, sol)
    plot_thrust(mdl, sol)
    plot_gimbal(mdl, sol)

    return sol, history
end
