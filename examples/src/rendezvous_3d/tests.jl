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

if isdefined(@__MODULE__, :LanguageServer)
    include("definition.jl")
    include("plots.jl")
end

using ECOS
using Printf
using Test
using Parser
using Utils
import Solvers

export ptr

function ptr()::Nothing

    # Problem definition
    N = 25
    mdl = RendezvousProblem()
    pbm = TrajectoryProblem(mdl)
    define_problem!(pbm, :ptr, N)

    # PTR algorithm parameters
    Nsub = 10
    iter_max = 30
    disc_method = IMPULSE
    wvc = 1e4
    wtr = 5e0
    ε_abs = -Inf
    ε_rel = 1e-3/100
    feas_tol = 5e-3
    q_tr = Inf
    q_exit = Inf
    solver = ECOS
    solver_options = Dict("verbose"=>0, "maxit"=>1000)
    pars = Solvers.PTR.Parameters(
        N, Nsub, iter_max, disc_method, wvc, wtr, ε_abs,
        ε_rel, feas_tol, q_tr, q_exit, solver,
        solver_options)

    # Solve the trajectory generation problem
    ptr = Solvers.PTR.create(pars, pbm)
    sol, history = Solvers.PTR.solve(ptr)

    @test sol.status == @sprintf("%s", SCP_SOLVED)

    # Make plots
    plot_trajectory_2d(mdl, sol)
    plot_state_timeseries(mdl, sol)
    plot_inputs(mdl, sol, history)
    plot_convergence(history, "rendezvous_3d")

    return nothing
end # function
