""" PTR algorithm data structures and methods.

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
this program.  If not, see <https://www.gnu.org/licenses/>. """

using LinearAlgebra
using JuMP
using Printf
using ..Utils
using ..Parser

import ..ST,
    ..RealTypes,
    ..IntRange,
    ..RealVector,
    ..RealMatrix,
    ..Trajectory,
    ..Objective,
    ..VarArgBlk,
    ..CstArgBlk,
    ..DLTV

import ..SCPParameters,
    ..SCPSubproblem, ..SCPSubproblemSolution, ..SCPProblem, ..SCPSolution, ..SCPHistory

import ..warm_start
import ..discretize!
import ..compute_original_cost!,
    ..add_dynamics!,
    ..add_convex_state_constraints!,
    ..add_convex_input_constraints!,
    ..add_nonconvex_constraints!,
    ..add_bcs!
import ..solve_subproblem!,
    ..solution_deviation, ..unsafe_solution, ..overhead!, ..save!, ..get_time

const CLP = ConicLinearProgram
const Variable = ST.Variable
const Optional = ST.Optional
const OptVarArgBlk = Optional{VarArgBlk}

export Parameters, create, solve

""" Structure holding the PTR algorithm parameters. """
mutable struct Parameters <: SCPParameters
    N::Int               # Number of temporal grid nodes
    Nsub::Int            # Number of subinterval integration time nodes
    iter_max::Int        # Maximum number of iterations
    disc_method::DiscretizationType # The discretization method
    wvc::RealTypes       # Virtual control weight
    wtr::RealTypes       # Trust region weight
    ε_abs::RealTypes     # Absolute convergence tolerance
    ε_rel::RealTypes     # Relative convergence tolerance
    feas_tol::RealTypes  # Dynamic feasibility tolerance
    q_tr::RealTypes      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::RealTypes    # Stopping criterion norm
    solver::Module       # The numerical solver to use for the subproblems
    solver_opts::Dict{String,Any} # Numerical solver options
end # struct

""" PTR subproblem solution. """
mutable struct SubproblemSolution <: SCPSubproblemSolution
    iter::Int             # PTR iteration number
    # >> Discrete-time rajectory <<
    xd::RealMatrix        # States
    ud::RealMatrix        # Inputs
    p::RealVector         # Parameter vector
    # >> Virtual control terms <<
    vd::RealMatrix        # Dynamics virtual control
    vs::RealMatrix        # Nonconvex constraints virtual control
    vic::RealVector       # Initial conditions virtual control
    vtc::RealVector       # Terminal conditions virtual control
    # >> Cost values <<
    J::RealTypes          # The original cost
    J_tr::RealTypes       # The trust region penalty
    J_vc::RealTypes       # The virtual control penalty
    J_aug::RealTypes      # Overall cost
    # >> Trajectory properties <<
    ηx::RealVector        # State trust region radii
    ηu::RealVector        # Input trust region radii
    ηp::RealTypes         # Parameter trust region radii
    status::ST.ExitStatus # Numerical optimizer exit status
    feas::Bool            # Dynamic feasibility flag
    defect::RealMatrix    # "Defect" linearization accuracy metric
    deviation::RealTypes  # Deviation from reference trajectory
    improv_rel::RealTypes # Relative cost improvement
    unsafe::Bool          # Indicator that the solution is unsafe to use
    dyn::DLTV             # The dynamics
    bay::Dict             # Storage bay for user-set values during callback
end # struct

""" Subproblem definition for the convex numerical optimizer. """
mutable struct Subproblem <: SCPSubproblem
    iter::Int            # PTR iteration number
    prg::ConicProgram    # The optimization problem object
    algo::String         # SCP and convex algorithms used
    # >> Algorithm parameters <<
    def::SCPProblem      # The PTR problem definition
    # >> Reference and solution trajectories <<
    sol::Union{SubproblemSolution,Missing} # Solution trajectory
    ref::Union{SubproblemSolution,Missing} # Reference trajectory
    # >> Cost function <<
    J::Objective        # The original convex cost function
    J_tr::Objective     # The virtual control penalty
    J_vc::Objective     # The virtual control penalty
    J_aug::Objective    # Overall cost function
    # >> Physical variables <<
    x::VarArgBlk        # Discrete-time states
    u::VarArgBlk        # Discrete-time inputs
    p::VarArgBlk        # Parameters
    # >> Virtual control (never scaled) <<
    vd::VarArgBlk       # Dynamics virtual control
    vs::OptVarArgBlk    # Nonconvex constraints virtual control
    vic::OptVarArgBlk   # Initial conditions virtual control
    vtc::OptVarArgBlk   # Terminal conditions virtual control
    # >> Trust region <<
    ηx::VarArgBlk       # State trust region radii
    ηu::VarArgBlk       # Input trust region radii
    ηp::VarArgBlk       # Parameter trust region radii
    # >> Statistics <<
    timing::Dict{Symbol,RealTypes} # Runtime profiling
end # struct

"""
    create(pars, traj)

Construct the PTR problem definition.

# Arguments
- `pars`: PTR algorithm parameters.
- `traj`: the underlying trajectory optimization problem.

# Returns
- `pbm`: the problem structure ready for being solved by PTR.
"""
function create(pars::Parameters, traj::TrajectoryProblem)::SCPProblem

    # Default progress table columns
    default_columns = [
        # Iteration count
        (:iter, "k", "%d", 2),
        # Solver status
        (:status, "status", "%s", 8),
        # Maximum dynamics virtual control element
        (:maxvd, "vd", "%.0e", 5),
        # Maximum constraints virtual control element
        (:maxvs, "vs", "%.0e", 5),
        # Maximum boundary conditions virtual control element
        (:maxvbc, "vbc", "%.0e", 5),
        # Overall cost (including penalties)
        (:cost, "J", "%.2e", 9),
        # Cost improvement (percent)
        (:ΔJ, "ΔJ %", "%s", 9),
        # Maximum deviation in state
        (:dx, "Δx", "%.0e", 5),
        # Maximum deviation in input
        (:du, "Δu", "%.0e", 5),
        # Maximum deviation in input
        (:dp, "Δp", "%.0e", 5),
        # Stopping criterion deviation measurement
        (:δ, "δ", "%.0e", 5),
        # Dynamic feasibility flag (true or false)
        (:dynfeas, "dyn", "%s", 3),
        # Maximum state trust region size
        (:trx_max, "ηx", "%.2f", 5),
        # Maximum input trust region size
        (:tru_max, "ηu", "%.2f", 5),
        # Parameter trust region size
        (:trp, "ηp", "%.2f", 5),
    ]

    # User-defined extra columns
    user_columns =
        Tuple{Symbol,String,String,Int}[tuple(col[1:4]...) for col in traj.table_cols]

    all_columns = [default_columns; user_columns]

    table = ST.Table(all_columns)

    pbm = SCPProblem(pars, traj, table)

    return pbm
end

"""
    Subproblem(pbm, iter[, ref])

Constructor for an empty convex optimization subproblem.

No cost or constraints. Just the decision variables and empty associated
parameters.

# Arguments
- `pbm`: the PTR problem being solved.
- `iter`: PTR iteration number.
- `ref`: (optional) the reference trajectory.

# Returns
- `spbm`: the subproblem structure.
"""
function Subproblem(
    pbm::SCPProblem,
    iter::Int,
    ref::Union{SubproblemSolution,Missing} = missing,
)::Subproblem

    # Statistics
    timing = Dict(:formulate => get_time(), :total => get_time())

    # Convenience values
    pars = pbm.pars
    scale = pbm.common.scale
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    N = pbm.pars.N
    _E = pbm.common.E

    # Optimization problem handle
    solver = pars.solver
    solver_opts = pars.solver_opts
    prg = ConicProgram(pbm.traj; solver = solver.Optimizer, solver_options = solver_opts)
    cvx_algo = string(pars.solver)
    algo = @sprintf("PTR (backend: %s)", cvx_algo)

    sol = missing # No solution associated yet with the subproblem

    # Cost
    J = missing
    J_tr = missing
    J_vc = missing
    J_aug = missing

    # Decision variables
    x = @new_variable(prg, (nx, N), "x")
    u = @new_variable(prg, (nu, N), "u")
    p = @new_variable(prg, np, "p")
    Sx = diag(scale.Sx)
    Su = diag(scale.Su)
    Sp = diag(scale.Sp)
    @scale(x, Sx, scale.cx)
    @scale(u, Su, scale.cu)
    @scale(p, Sp, scale.cp)

    # Virtual controls
    vd = @new_variable(prg, (size(_E, 2), N - 1), "vd")
    vs = nothing
    vic = nothing
    vtc = nothing

    # Trust region radii
    ηx = @new_variable(prg, N, "ηx")
    ηu = @new_variable(prg, N, "ηu")
    ηp = @new_variable(prg, "ηp")

    spbm = Subproblem(
        iter,
        prg,
        algo,
        pbm,
        sol,
        ref,
        J,
        J_tr,
        J_vc,
        J_aug,
        x,
        u,
        p,
        vd,
        vs,
        vic,
        vtc,
        ηx,
        ηu,
        ηp,
        timing,
    )

    return spbm
end

"""
    SubproblemSolution(x, u, p, iter, pbm)

Construct a subproblem solution from a discrete-time trajectory.

This leaves parameters of the solution other than the passed discrete-time
trajectory unset.

# Arguments
- `x`: discrete-time state trajectory.
- `u`: discrete-time input trajectory.
- `p`: parameter vector.
- `iter`: PTR iteration number.
- `pbm`: the PTR problem definition.

# Returns
- `subsol`: subproblem solution structure.
"""
function SubproblemSolution(
    x::RealMatrix,
    u::RealMatrix,
    p::RealVector,
    iter::Int,
    pbm::SCPProblem,
)::SubproblemSolution

    # Parameters
    N = pbm.pars.N
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    nv = size(pbm.common.E, 2)
    disc = pbm.pars.disc_method

    # Uninitialized parts
    ηx = fill(NaN, N)
    ηu = fill(NaN, N)
    ηp = NaN
    status = MOI.OPTIMIZE_NOT_CALLED
    feas = false
    defect = fill(NaN, nx, N - 1)
    deviation = NaN
    improv_rel = NaN
    unsafe = false
    dyn = DLTV(nx, nu, np, nv, N, disc)
    bay = Dict()

    vd = RealMatrix(undef, 0, N)
    vs = RealMatrix(undef, 0, N)
    vic = RealVector(undef, 0)
    vtc = RealVector(undef, 0)

    J = NaN
    J_tr = NaN
    J_vc = NaN
    J_aug = NaN
    J_aug = NaN

    subsol = SubproblemSolution(
        iter,
        x,
        u,
        p,
        vd,
        vs,
        vic,
        vtc,
        J,
        J_tr,
        J_vc,
        J_aug,
        ηx,
        ηu,
        ηp,
        status,
        feas,
        defect,
        deviation,
        improv_rel,
        unsafe,
        dyn,
        bay,
    )

    # Compute the DLTV dynamics around this solution
    discretize!(subsol, pbm)

    return subsol
end

"""
    SubproblemSolution(spbm)

Construct subproblem solution from a subproblem object.

Expects that the subproblem argument is a solved subproblem (i.e. one to which
numerical optimization has been applied).

# Arguments
- `spbm`: the subproblem structure.

# Returns
- `sol`: subproblem solution.
"""
function SubproblemSolution(spbm::Subproblem)::SubproblemSolution
    # Extract the discrete-time trajectory
    x = value(spbm.x)
    u = value(spbm.u)
    p = value(spbm.p)

    # Form the partly uninitialized subproblem
    sol = SubproblemSolution(x, u, p, spbm.iter, spbm.def)

    # Save the virtual control values and penalty terms
    sol.vd = value(spbm.vd)
    if !isnothing(spbm.vs)
        sol.vs = value(spbm.vs)
    end
    if !isnothing(spbm.vic)
        sol.vic = value(spbm.vic)
    end
    if !isnothing(spbm.vtc)
        sol.vtc = value(spbm.vtc)
    end

    # Save the optimal cost values
    sol.J = value(spbm.J)
    sol.J_tr = value(spbm.J_tr)
    sol.J_vc = value(spbm.J_vc)
    sol.J_aug = value(spbm.J_aug)

    # Save the trust region radii
    sol.ηx = value(spbm.ηx)
    sol.ηu = value(spbm.ηu)
    sol.ηp = value(spbm.ηp)[1]

    return sol
end

"""
    ptr_solve(pbm[, warm])

Solve the optimal control problem using the penalized trust region (PTR)
method.

# Arguments
- `pbm`: the trajectory problem to be solved.
- `warm`: (optional) warm start solution.

# Returns
- `sol`: the PTR solution structur.
- `history`: PTR iteration data history.
"""
function solve(
    pbm::SCPProblem,
    warm::Union{Nothing,SCPSolution} = nothing,
)::Tuple{Union{SCPSolution,Nothing},SCPHistory}
    # ..:: Initialize ::..

    if isnothing(warm)
        ref = generate_initial_guess(pbm)
    else
        ref = warm_start(pbm, warm, SubproblemSolution)
    end

    history = SCPHistory()

    callback_fun! = pbm.traj.callback!
    user_callback = !isnothing(callback_fun!)

    # ..:: Iterate ::..

    k = 1 # Iteration counter
    while true
        # Construct the subproblem
        spbm = Subproblem(pbm, k, ref)

        add_dynamics!(spbm)
        add_convex_state_constraints!(spbm)
        add_convex_input_constraints!(spbm)
        add_nonconvex_constraints!(spbm)
        add_bcs!(spbm)
        add_trust_region!(spbm)
        add_cost!(spbm)

        save!(history, spbm)

        try
            # Solve the subproblem
            solve_subproblem!(spbm)

            # "Emergency exit" the PTR loop if something bad happened
            # (e.g. numerical problems)
            if unsafe_solution(spbm)
                print_info(spbm)
                break
            end

            # Check stopping criterion
            stop = check_stopping_criterion!(spbm)

            # Run a user-defined callback
            if user_callback
                user_acted = callback_fun!(spbm)
            end

            # Stop iterating if stopping criterion triggered **and** user did
            # not modify anything in the callback
            if stop && !(user_callback && user_acted)
                print_info(spbm)
                break
            end

            # Update reference trajectory
            ref = spbm.sol
        catch e
            isa(e, SCPError) || rethrow(e)
            print_info(spbm, e)
            break
        end

        # Print iteration info
        print_info(spbm)

        # Stop at maximum iterations
        k += 1
        if k > pbm.pars.iter_max
            break
        end
    end

    reset(pbm.common.table)

    # ..:: Save solution ::..
    sol = SCPSolution(history)

    return sol, history
end

"""
    generate_initial_guess(pbm)

Compute the initial trajectory guess.

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to an SubproblemSolution structure.

# Arguments
- `pbm`: the PTR problem structure.

# Returns
- `guess`: the initial guess.
"""
function generate_initial_guess(pbm::SCPProblem)::SubproblemSolution

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    guess = SubproblemSolution(x, u, p, 0, pbm)

    return guess
end

"""
    add_trust_region!(spbm)

Add trust region constraint to the subproblem.

# Arguments
- `spbm`: the subproblem definition.
"""
function add_trust_region!(spbm::Subproblem)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    q = spbm.def.pars.q_tr
    scale = spbm.def.common.scale
    prg = spbm.prg
    x = spbm.x
    u = spbm.u
    p = spbm.p
    xh_ref = scale.iSx * (spbm.ref.xd .- scale.cx)
    uh_ref = scale.iSu * (spbm.ref.ud .- scale.cu)
    ph_ref = scale.iSp * (spbm.ref.p - scale.cp)
    ηx = spbm.ηx
    ηu = spbm.ηu
    ηp = spbm.ηp

    q2cone = Dict(1 => L1, 2 => SOC, 4 => SOC, Inf => LINF)
    cone = q2cone[q]

    # >> Parameter trust region <<
    dp_lq = @new_variable(prg, "dp_lq")

    @add_constraint(
        prg,
        cone,
        "parameter_trust_region",
        (p, dp_lq),
        begin
            local p, dp_lq = arg
            local ph = scale.iSp * (p - scale.cp)
            local dp = ph - ph_ref
            vcat(dp_lq, dp)
        end
    )

    if q == 4
        wp = @new_variable(prg, "wp")
        @add_constraint(
            prg,
            SOC,
            "parameter_trust_region",
            (wp, dp_lq),
            begin
                local wp, dp_lq = arg
                vcat(wp, dp_lq)
            end
        )
        @add_constraint(
            prg,
            GEOM,
            "parameter_trust_region",
            (wp, ηp),
            begin
                local wp, ηp = arg
                vcat(wp, ηp, 1)
            end
        )
    else
        @add_constraint(
            prg,
            NONPOS,
            "parameter_trust_region",
            (ηp, dp_lq),
            begin
                local ηp, dp_lq = arg
                dp_lq - ηp
            end
        )
    end

    # State trust regions
    dx_lq = @new_variable(prg, N, "dx_lq")

    for k = 1:N
        @add_constraint(
            prg,
            cone,
            "state_trust_region",
            (x[:, k], dx_lq[k]),
            begin
                local xk, dxk_lq = arg
                local xhk = scale.iSx * (xk - scale.cx)
                local dxk = xhk - xh_ref[:, k]
                vcat(dxk_lq, dxk)
            end
        )
        if q == 4
            # State
            wx = @new_variable(prg, "wx")
            @add_constraint(
                prg,
                SOC,
                "state_trust_region",
                (wx, dx_lq[k]),
                begin
                    local wx, dxk_lq = arg
                    vcat(wx, dxk_lq)
                end
            )
            @add_constraint(
                prg,
                GEOM,
                "state_trust_region",
                (wx, ηx[k]),
                begin
                    local wx, ηxk = arg
                    vcat(wx, ηxk, 1)
                end
            )
        else
            # State
            @add_constraint(
                prg,
                NONPOS,
                "state_trust_region",
                (ηx[k], dx_lq[k]),
                begin
                    local ηxk, dxk_lq = arg
                    dxk_lq - ηxk
                end
            )
        end
    end

    # Input trust regions
    du_lq = @new_variable(prg, N, "du_lq")

    for k = 1:N
        @add_constraint(
            prg,
            cone,
            "input_trust_region",
            (u[:, k], du_lq[k]),
            begin
                local uk, duk_lq = arg
                local uhk = scale.iSu * (uk - scale.cu)
                local duk = uhk - uh_ref[:, k]
                vcat(duk_lq, duk)
            end
        )
        if q == 4
            wu = @new_variable(prg, "wu")
            @add_constraint(
                prg,
                SOC,
                "input_trust_region",
                (wu, du_lq[k]),
                begin
                    local wu, duk_lq = arg
                    vcat(wu, duk_lq)
                end
            )
            @add_constraint(
                prg,
                GEOM,
                "input_trust_region",
                (wu, ηu[k]),
                begin
                    local wu, ηuk = arg
                    vcat(wu, ηuk, 1)
                end
            )
        else
            @add_constraint(
                prg,
                NONPOS,
                "input_trust_region",
                (ηu[k], du_lq[k]),
                begin
                    local ηuk, duk_lq = arg
                    duk_lq - ηuk
                end
            )
        end
    end

    return nothing
end

"""
    add_cost!(spbm)

Define the subproblem cost function.

# Arguments
- `spbm`: the subproblem definition.
"""
function add_cost!(spbm::Subproblem)::Nothing

    # Compute the cost components
    spbm.J = compute_original_cost!(spbm)
    compute_trust_region_penalty!(spbm)
    compute_virtual_control_penalty!(spbm)

    spbm.J_aug = cost(spbm.prg)

    return nothing
end

"""
    compute_trust_region_penalty!(spbm)

Compute the subproblem cost trust region penalty term.

# Arguments
- `spbm`: the subproblem definition.
"""
function compute_trust_region_penalty!(spbm::Subproblem)::Nothing

    # Variables and parameters
    t = spbm.def.common.t_grid
    wtr = spbm.def.pars.wtr
    prg = spbm.prg
    ηx = spbm.ηx
    ηu = spbm.ηu
    ηp = spbm.ηp

    spbm.J_tr = @add_cost(prg, (ηx, ηu, ηp), begin
        local ηx, ηu, ηp = arg
        wtr * (trapz(ηx, t) + trapz(ηu, t) + ηp[1])
    end)

    return nothing
end

"""
    compute_virtual_control_penalty!(spbm)

Compute the subproblem cost virtual control penalty term.

# Arguments
- `spbm`: the subproblem definition.
"""
function compute_virtual_control_penalty!(spbm::Subproblem)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    wvc = spbm.def.pars.wvc
    t = spbm.def.common.t_grid
    E = spbm.ref.dyn.E
    prg = spbm.prg
    vd = spbm.vd
    vs = spbm.vs
    vic = spbm.vic
    vtc = spbm.vtc

    # Virtual control penalty
    P = @new_variable(prg, N, "P")
    Pf = @new_variable(prg, 2, "Pf")
    if !isnothing(vs)
        for k = 1:N
            if k < N
                @add_constraint(
                    prg,
                    L1,
                    "vd_vs_penalty",
                    (P[k], vd[:, k], vs[:, k]),
                    begin
                        local Pk, vdk, vsk = arg
                        vcat(Pk, E[:, :, k] * vdk, vsk)
                    end
                )
            else
                @add_constraint(
                    prg,
                    L1,
                    "vd_vs_penalty",
                    (P[k], vs[:, k]),
                    begin
                        local Pk, vsk = arg
                        vcat(Pk, vsk)
                    end
                )
            end
        end
    else
        for k = 1:N
            if k < N
                @add_constraint(
                    prg,
                    L1,
                    "vd_vs_penalty",
                    (P[k], vd[:, k]),
                    begin
                        local Pk, vdk = arg
                        vcat(Pk, E[:, :, k] * vdk)
                    end
                )
            else
                @add_constraint(prg, ZERO, "vd_vs_penalty", (P[k],), begin
                    local Pk, = arg
                    Pk
                end)
            end
        end
    end

    # Initial condition relaxation penalty
    if !isnothing(vic)
        @add_constraint(prg, L1, "vic_penalty", (Pf[1], vic), begin
            local Pf1, vic = arg
            vcat(Pf1, vic)
        end)
    else
        @add_constraint(prg, ZERO, "vic_penalty", (Pf[1]), begin
            local Pf1, = arg
            Pf1
        end)
    end

    # Terminal condition relaxation penalty
    if !isnothing(vtc)
        @add_constraint(prg, L1, "vtc_penalty", (Pf[2], vtc), begin
            local Pf2, vtc = arg
            vcat(Pf2, vtc)
        end)
    else
        @add_constraint(prg, ZERO, "vtc_penalty", (Pf[2],), begin
            local Pf2, = arg
            Pf2
        end)
    end

    spbm.J_vc = @add_cost(prg, (P, Pf), begin
        local P, Pf = arg
        wvc * (trapz(P, t) + sum(Pf))
    end)

    return nothing
end

"""
    check_stopping_criterion!(spbm)

Check if stopping criterion is triggered.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `stop`: true if stopping criterion holds.
"""
function check_stopping_criterion!(spbm::Subproblem)::Bool

    # Extract values
    pbm = spbm.def
    ref = spbm.ref
    sol = spbm.sol
    ε_abs = pbm.pars.ε_abs
    ε_rel = pbm.pars.ε_rel

    # Compute solution deviation from reference
    sol.deviation = solution_deviation(spbm)

    # Check predicted cost improvement
    J_ref = ref.J_aug
    J_sol = sol.J_aug
    sol.improv_rel = (J_ref - J_sol) / abs(J_ref)

    # Compute stopping criterion
    stop = (
        spbm.iter > 1 &&
        (sol.feas && (abs(sol.improv_rel) <= ε_rel || sol.deviation <= ε_abs))
    )

    return stop
end

"""
    print_info(spbm[, err])

Print command line info message.

# Arguments
- `spbm`: the subproblem that was solved.
- `err`: an PTR-specific error message.
"""
function print_info(spbm::Subproblem, err::Union{Nothing,SCPError} = nothing)::Nothing

    # Convenience variables
    sol = spbm.sol
    traj = spbm.def.traj
    ref = spbm.ref
    table = spbm.def.common.table

    if !isnothing(err)
        @printf "%s, exiting\n" err.msg
    elseif unsafe_solution(sol)
        @printf "unsafe solution (%s), exiting\n" sol.status
    else
        # Preprocess values
        scale = spbm.def.common.scale
        xh = scale.iSx * (sol.xd .- scale.cx)
        uh = scale.iSu * (sol.ud .- scale.cu)
        ph = scale.iSp * (sol.p - scale.cp)
        xh_ref = scale.iSx * (spbm.ref.xd .- scale.cx)
        uh_ref = scale.iSu * (spbm.ref.ud .- scale.cu)
        ph_ref = scale.iSp * (spbm.ref.p - scale.cp)
        max_dxh = norm(xh - xh_ref, Inf)
        max_duh = norm(uh - uh_ref, Inf)
        max_dph = norm(ph - ph_ref, Inf)
        status = @sprintf "%s" sol.status
        status = status[1:min(8, length(status))]
        ΔJ = improvement_percent(sol.J_aug, ref.J_aug)
        ηx_max = maximum(sol.ηx)
        ηu_max = maximum(sol.ηu)
        ηp = sol.ηp

        # Associate values with columns
        assoc = Dict(
            :iter => spbm.iter,
            :status => status,
            :maxvd => norm(sol.vd, Inf),
            :maxvs => norm(sol.vs, Inf),
            :maxvbc => norm([sol.vic; sol.vtc], Inf),
            :cost => sol.J_aug,
            :ΔJ => ΔJ,
            :dx => max_dxh,
            :du => max_duh,
            :dp => max_dph,
            :δ => sol.deviation,
            :dynfeas => sol.feas ? "T" : "F",
            :trx_max => ηx_max,
            :tru_max => ηu_max,
            :trp => ηp,
        )

        # Set user-defined columns
        for col in traj.table_cols
            id, col_value = col[1], col[end]
            assoc[id] = col_value(sol.bay)
        end

        print(assoc, table)
    end

    overhead!(spbm)

    return nothing
end
