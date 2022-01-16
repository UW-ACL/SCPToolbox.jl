#= Tests for spacecraft rendezvous with discrete logic.

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
    pars = PTR.Parameters(
        N, Nsub, iter_max, disc_method, wvc, wtr, ε_abs,
        ε_rel, feas_tol, q_tr, q_exit, solver,
        solver_options)

    test_single(mdl, pbm, pars)
    test_runtime(mdl, pbm, pars)
    test_homotopy_update(mdl, pbm, pars)

    return nothing
end

"""
    test_single(pbm, pars)

Compute a single trajectory.

# Arguments
- `mdl`: the rendezvous problem definition.
- `pbm`: the rendezvous trajectory problem.
- `pars`: the rendezvous trajectory problem.

# Returns
- `sol`: the trajectory solution.
- `history`: the iterate history.
"""
function test_single(mdl::RendezvousProblem,
                     pbm::TrajectoryProblem,
                     pars::PTR.Parameters)::Tuple{SCPSolution,
                                                  SCPHistory}

    test_heading("Single trajectory")

    # Create problem
    ptr = PTR.create(pars, pbm)
    reset_homotopy(pbm)

    # Solve problem
    sol, history = PTR.solve(ptr)

    @test sol.status == @sprintf("%s", SCP_SOLVED)

    # Make plots
    plot_trajectory_2d(mdl, sol)
    plot_trajectory_2d(mdl, sol; attitude=true)
    plot_state_timeseries(mdl, sol)
    plot_inputs(mdl, sol, history)
    plot_inputs(mdl, sol, history; quad="D")
    plot_cost_evolution(mdl, history)

    return sol, history
end

"""
    test_single(pbm, pars)

Run the algorithm several times and plot runtime statistics.

# Arguments
- `mdl`: the rendezvous problem definition.
- `pbm`: the rendezvous trajectory problem.
- `pars`: the rendezvous trajectory problem.

# Returns
- `history_list`: vector of iterate histories for each trial.
"""
function test_runtime(mdl::RendezvousProblem,
                      pbm::TrajectoryProblem,
                      pars::PTR.Parameters)::Vector{SCPHistory}

    test_heading("Runtime statistics")

    num_trials=20

    history_list = Vector{SCPHistory}(undef, num_trials)

    for trial = 1:num_trials

        # Create new problem
        ptr_pbm = PTR.create(pars, pbm)
        reset_homotopy(pbm)

        @printf("Trial %d/%d\n", trial, num_trials)

        # Suppress output
        real_stdout = stdout
        (rd, wr) = redirect_stdout()

        # Run algorithm
        sol, history_list[trial] = PTR.solve(ptr_pbm)

        @test sol.status == @sprintf("%s", SCP_SOLVED)

        redirect_stdout(real_stdout) # Revert to normal output
    end

    plot_convergence(history_list, "rendezvous_3d",
                     options=fig_opts, xlabel="\$\\ell\$",
                     horizontal=true)

    return history_list
end

"""
    test_homotopy_update(pbm, pars)

Test a sweep of homotopy update thresholds.

# Arguments
- `mdl`: the rendezvous problem definition.
- `pbm`: the rendezvous trajectory problem.
- `pars`: the rendezvous trajectory problem.

# Returns
- `β_sweep`: vector of homotopy update thresholds that were tested.
- `sol_list`: vector of trajectory solutions that were obtained for each
  setting.
"""
function test_homotopy_update(mdl::RendezvousProblem,
                              pbm::TrajectoryProblem,
                              pars::PTR.Parameters)::Tuple{
                                  Vector{Float64},
                                  Vector{SCPSolution}}

    test_heading("Homotopy update sweep")

    resol = 20
    β_sweep = collect(LinRange(0.1, 50, resol))/100

    sol_list = Vector{SCPSolution}(undef, resol)

    for i = 1:resol

        mdl.traj.β = β_sweep[i]

        # Create new problem
        ptr_pbm = PTR.create(pars, pbm)
        reset_homotopy(pbm)

        @printf("(%d/%d) β = %.2e\n", i, resol, mdl.traj.β)

        # Suppress output
        real_stdout = stdout
        (rd, wr) = redirect_stdout()

        # Run algorithm
        sol_list[i], _ = PTR.solve(ptr_pbm)

        @test sol_list[i].status == @sprintf("%s", SCP_SOLVED)

        redirect_stdout(real_stdout) # Revert to normal output
    end

    plot_homotopy_threshold_sweep(mdl, β_sweep, sol_list)

    return β_sweep, sol_list
end

"""
    reset_homotopy(pbm)

Reset the homotopy value back to the initial one.

# Arguments
- `pbm`: the rendezvous trajectory problem.
"""
function reset_homotopy(pbm::TrajectoryProblem)::Nothing
    pbm.mdl.traj.hom = pbm.mdl.traj.hom_grid[1] # Reset homotopy
    return nothing
end
