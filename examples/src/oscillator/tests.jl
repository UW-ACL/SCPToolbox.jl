#= Tests for forced harmonic oscillator with input deadband.

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

function ptr()::Nothing

    # Problem definition
    N = 30
    mdl = OscillatorProblem(N)
    pbm = TrajectoryProblem(mdl)
    define_problem!(pbm, :ptr)

    # PTR algorithm parameters
    Nsub = 10
    iter_max = 10
    disc_method = FOH
    wvc = 1e2
    wtr = 1e-3
    ε_abs = -Inf#1e-5
    ε_rel = 1e-3/100
    feas_tol = 5e-3
    q_tr = Inf
    q_exit = Inf
    solver = ECOS
    solver_options = Dict("verbose"=>0)
    pars = Solvers.PTR.Parameters(
        N, Nsub, iter_max, disc_method, wvc, wtr, ε_abs, ε_rel,
        feas_tol, q_tr, q_exit, solver, solver_options)

    # Homotopy parameters
    Nhom = 10
    hom_κ1 = Homotopy(1e-8)
    hom_grid = LinRange(0.0, 1.0, Nhom)

    # Solve the trajectory generation problem
    ptr_pbm = Solvers.PTR.create(pars, pbm)
    sols, historys = [], []
    for i = 1:Nhom
        mdl.traj.κ1 = hom_κ1(hom_grid[i])
        warm = (i==1) ? nothing : sols[end]

        @printf("[%d/%d] Homotopy (κ=%.2e)\n", i, Nhom, mdl.traj.κ1)

        sol_i, history_i = Solvers.PTR.solve(ptr_pbm, warm)

        push!(sols, sol_i)
        push!(historys, history_i)
    end
    sol = sols[end]
    history = historys[end]

    @test sol.status == @sprintf("%s", SCP_SOLVED)

    # Make plots
    plot_timeseries(mdl, sol, history)
    plot_deadband(mdl, sol)
    plot_convergence(history, "oscillator")

    return nothing
end
