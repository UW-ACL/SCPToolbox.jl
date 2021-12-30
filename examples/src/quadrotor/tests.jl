"""
Tests for quadrotor obstacle avoidance.

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
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

using ECOS
using Printf
using Test
using Parser
using Utils
using Solvers

export scvx
export gusto

const SCvx = Solvers.SCvx
const GuSTO = Solvers.GuSTO

function scvx()::Nothing

    # Problem definition
    mdl = QuadrotorProblem()
    pbm = TrajectoryProblem(mdl)
    define_problem!(pbm, :scvx)

    # SCvx algorithm parameters
    N = 30
    Nsub = 15
    iter_max = 15
    disc_method = FOH
    λ = 30.0
    ρ_0 = 0.0
    ρ_1 = 0.1
    ρ_2 = 0.7
    β_sh = 2.0
    β_gr = 2.0
    η_init = 1.0
    η_lb = 1e-3
    η_ub = 10.0
    ε_abs = 0#1e-5
    ε_rel = 0#0.01/100
    feas_tol = 1e-3
    q_tr = Inf
    q_exit = Inf
    solver = ECOS
    solver_options = Dict("verbose"=>0)
    pars = SCvx.Parameters(
        N, Nsub, iter_max, disc_method, λ, ρ_0, ρ_1, ρ_2, β_sh, β_gr,
        η_init, η_lb, η_ub, ε_abs, ε_rel, feas_tol, q_tr, q_exit, solver,
        solver_options)

    # Solve multiple times to gather statistics
    run_trials(mdl, pbm, pars, SCvx)

end

function gusto()::Nothing

    # Problem definition
    mdl = QuadrotorProblem()
    pbm = TrajectoryProblem(mdl)
    define_problem!(pbm, :scvx)

    # SCvx algorithm parameters
    N = 30
    Nsub = 15
    iter_max = 15
    disc_method = FOH
    λ_init = 1e4
    λ_max = 1e9
    ρ_0 = 0.1
    ρ_1 = 0.9
    β_sh = 2.0
    β_gr = 2.0
    γ_fail = 5.0
    η_init = 10.0
    η_lb = 1e-3
    η_ub = 10.0
    μ = 0.8
    iter_μ = 6
    ε_abs = 0#1e-5
    ε_rel = 0#0.01/100
    feas_tol = 1e-3
    pen = :quad
    hom = 100.0
    q_tr = Inf
    q_exit = Inf
    solver = ECOS
    solver_options = Dict("verbose"=>0)
    pars = GuSTO,Parameters(
        N, Nsub, iter_max, disc_method, λ_init, λ_max, ρ_0, ρ_1, β_sh, β_gr,
        γ_fail, η_init, η_lb, η_ub, μ, iter_μ, ε_abs, ε_rel, feas_tol, pen, hom,
        q_tr, q_exit, solver, solver_options)

    # Solve multiple times to gather statistics
    run_trials(mdl, pbm, pars, GuSTO)

end

"""
    run_trials(mdl, pbm, pars)

Solves the same problem multiple times in order to gather realiable runtime statistics.

# Parameters
- `mdl`: problem-specific data.
- `pbm`: the trajectory problem.
- `pars`: solution algorithm parameters.
- `solver`: the solver algorithm's module.
"""
function run_trials(
        mdl::FreeFlyerProblem,
        pbm::TrajectoryProblem,
        pars::T,
        solver::Module
)::Nothing where {T<:Solvers.SCPParameters}

    # Number of trials. All trials will give the same solution, but we need many to
    # plot statistically meaningful timing results
    num_trials = 100

    sol_list = Vector{SCPSolution}(undef, num_trials)
    history_list = Vector{SCPHistory}(undef, num_trials)

    for trial = 1:num_trials
        local pbm = solver.create(pars, pbm)
        @printf("Trial %d/%d\n", trial, num_trials)
        if trial>1
            # Suppress output
            real_stdout = stdout
            (rd, wr) = redirect_stdout()
        end
        sol_list[trial], history_list[trial] = solver.solve(pbm)
        @assert sol_list[trial].status == @sprintf("%s", SCP_SOLVED)
        if trial>1
            redirect_stdout(real_stdout)
        end
    end

    # Save one solution instance - for plotting a single trial
    sol = sol_list[end]
    history = history_list[end]

    # Make plots
    plot_trajectory_history(mdl, history)
    plot_final_trajectory(mdl, sol)
    plot_input_norm(mdl, sol)
    plot_tilt_angle(mdl, sol)
    plot_convergence(history_list, "quadrotor")

    return Nothing
end
