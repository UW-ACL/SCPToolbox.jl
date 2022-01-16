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
export scvx

const PTR = Solvers.PTR
const SCvx = Solvers.SCvx

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

    test_single(mdl, pbm, pars, PTR)

    return nothing
end

function scvx()::Nothing

    # Problem definition
    mdl = StarshipProblem()
    pbm = TrajectoryProblem(mdl)
    define_problem!(pbm, :scvx)

    # PTR algorithm parameters
    N = 31
    Nsub = 100
    iter_max = 100
    disc_method = FOH
    λ = 5e2
    ρ_0 = 0.0
    ρ_1 = 0.1
    ρ_2 = 0.7
    β_sh = 2.0
    β_gr = 2.0
    η_init = 1.0
    η_lb = 1e-8
    η_ub = 10.0
    ε_abs = 1e-5
    ε_rel = 0.01/100
    feas_tol = 5e-3
    q_tr = Inf
    q_exit = Inf
    solver = ECOS
    solver_options = Dict("verbose"=>0, "maxit"=>1000)
    pars = SCvx.Parameters(
        N, Nsub, iter_max, disc_method, λ, ρ_0, ρ_1, ρ_2, β_sh, β_gr,
        η_init, η_lb, η_ub, ε_abs, ε_rel, feas_tol, q_tr, q_exit, solver,
        solver_options)

    test_single(mdl, pbm, pars, SCvx)

    return nothing
end

"""
    test_single(pbm, traj, pars, solver)

Compute a single trajectory.

# Arguments
- `mdl`: the starship parameters.
- `traj`: the trajectory problem definition.
- `pars`: the algorithm parameters.
- `solver`: the solver algorithm's module.
"""
function test_single(
        mdl::StarshipProblem,
        traj::TrajectoryProblem,
        pars::T,
        solver::Module
)::Nothing where {T<:Solvers.SCPParameters}

    test_heading("Single trajectory")

    # Create problem
    pbm = solver.create(pars, traj)

    # Solve problem
    sol, history = solver.solve(pbm)

    @test sol.status == @sprintf("%s", SCP_SOLVED)

    # Make plots
    plot_trajectory_history(mdl, history)
    plot_final_trajectory(mdl, sol)
    plot_velocity(mdl, sol)
    plot_thrust(mdl, sol)
    plot_gimbal(mdl, sol)

    return nothing
end
