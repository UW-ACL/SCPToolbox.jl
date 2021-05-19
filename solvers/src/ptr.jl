#= PTR algorithm data structures and methods.

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

LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("../../utils/src/Utils.jl")
    include("../../parser/src/Parser.jl")

    include("discretization.jl")
    include("scp.jl")

    using .Utils
    using .Utils.Types: improvement_percent
    using .Parser.ConicLinearProgram

    import .Parser.ConicLinearProgram: ConicProgram, ConvexCone, SupportedCone
    import .Parser.ConicLinearProgram: VariableArgumentBlock
    import .Parser.ConicLinearProgram: ConstantArgumentBlock
    import .Parser.ConicLinearProgram: ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
end

using LinearAlgebra
using JuMP
using Printf

using Utils
using Parser

import ..ST, ..RealTypes, ..IntRange, ..RealVector, ..RealMatrix, ..Trajectory,
    ..Objective, ..VarArgBlk, ..CstArgBlk, ..DLTV

import ..SCPParameters, ..SCPSubproblem, ..SCPSubproblemSolution, ..SCPProblem,
    ..SCPSolution, ..SCPHistory

import ..discretize!
import ..add_dynamics!, ..add_convex_state_constraints!,
    ..add_convex_input_constraints!, ..add_nonconvex_constraints!, ..add_bcs!
import ..solve_subproblem!, ..solution_deviation, ..unsafe_solution,
    ..overhead!, ..save!, ..get_time

const CLP = ConicLinearProgram #noerr
const Variable = ST.Variable
const Optional = ST.Optional
const OptVarArgBlk = Optional{VarArgBlk}

export Parameters, create, solve

#= Structure holding the PTR algorithm parameters. =#
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
    solver_opts::Dict{String, Any} # Numerical solver options
end # struct

#= PTR subproblem solution. =#
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

#= Subproblem definition in JuMP format for the convex numerical optimizer. =#
mutable struct Subproblem <: SCPSubproblem
    iter::Int            # PTR iteration number
    prg::ConicProgram    # The optimization problem object
    algo::String         # SCP and convex algorithms used
    # >> Algorithm parameters <<
    def::SCPProblem      # The PTR problem definition
    # >> Reference and solution trajectories <<
    sol::Union{SubproblemSolution, Missing} # Solution trajectory
    ref::Union{SubproblemSolution, Missing} # Reference trajectory
    # >> Cost function <<
    J::Objective        # The original convex cost function
    J_tr::Objective     # The virtual control penalty
    J_vc::Objective     # The virtual control penalty
    J_aug::Objective    # Overall cost function
    # >> Physical variables <<
    x::VarArgBlk        # Discrete-time states
    u::VarArgBlk        # Discrete-time inputs
    p::VarArgBlk        # Parameters
    q::CstArgBlk        # Constant parameters
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
    timing::Dict{Symbol, RealTypes} # Runtime profiling
end # struct

#= Construct the PTR problem definition.

Args:
    pars: PTR algorithm parameters.
    traj: the underlying trajectory optimization problem.

Returns:
    pbm: the problem structure ready for being solved by PTR. =#
function create(pars::Parameters,
                traj::TrajectoryProblem)::SCPProblem

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
        (:trp, "ηp", "%.2f", 5)]

    # User-defined extra columns
    user_columns = [tuple(col[1:4]...) for col in traj.table_cols]

    all_columns = [default_columns; user_columns]

    table = ST.Table(all_columns)

    pbm = SCPProblem(pars, traj, table)

    return pbm
end # function

#= Constructor for an empty convex optimization subproblem.

No cost or constraints. Just the decision variables and empty associated
parameters.

Args:
    pbm: the PTR problem being solved.
    iter: PTR iteration number.
    ref: (optional) the reference trajectory.

Returns:
    spbm: the subproblem structure. =#
function Subproblem(pbm::SCPProblem, iter::Int,
                    ref::Union{SubproblemSolution,
                               Missing}=missing)::Subproblem

    # Statistics
    timing = Dict(:formulate => get_time(), :total => get_time())

    # Convenience values
    pars = pbm.pars
    scale = pbm.common.scale
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    nq = pbm.traj.nq
    N = pbm.pars.N
    _E = pbm.common.E

    # Optimization problem handle
    solver = pars.solver
    solver_opts = pars.solver_opts
    prg = ConicProgram(pbm.traj;
                       solver=solver.Optimizer,
                       solver_options=solver_opts)
    cvx_algo = string(pars.solver)
    algo = @sprintf("PTR (backend: %s)", cvx_algo)

    sol = missing # No solution associated yet with the subproblem

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

    # Constant parameters
    q = @new_parameter(prg, nq, "q")
    if nq>0
        q .= pbm.traj.setq()
        Sq = diag(scale.Sq)
        @scale(q, Sq, scale.cq)
    end

    # Virtual controls
    vd = @new_variable(prg, (size(_E, 2), N-1), "vd")
    vs = nothing
    vic = nothing
    vtc = nothing

    # Trust region radii
    ηx = @new_variable(prg, N, "ηx")
    ηu = @new_variable(prg, N, "ηu")
    ηp = @new_variable(prg, "ηp")

    spbm = Subproblem(iter, prg, algo, pbm, sol, ref, J, J_tr, J_vc, J_aug,
                      x, u, p, q, vd, vs, vic, vtc, ηx, ηu, ηp, timing)

    return spbm
end # function

#= Construct a subproblem solution from a discrete-time trajectory.

This leaves parameters of the solution other than the passed discrete-time
trajectory unset.

Args:
    x: discrete-time state trajectory.
    u: discrete-time input trajectory.
    p: parameter vector.
    iter: PTR iteration number.
    pbm: the PTR problem definition.

Returns:
    subsol: subproblem solution structure. =#
function SubproblemSolution(
    x::RealMatrix,
    u::RealMatrix,
    p::RealVector,
    iter::Int,
    pbm::SCPProblem)::SubproblemSolution

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
    defect = fill(NaN, nx, N-1)
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

    subsol = SubproblemSolution(iter, x, u, p, vd, vs, vic, vtc, J,
                                J_tr, J_vc, J_aug, ηx, ηu, ηp, status,
                                feas, defect, deviation, improv_rel,
                                unsafe, dyn, bay)

    # Compute the DLTV dynamics around this solution
    discretize!(subsol, pbm)

    return subsol
end # function

#= Construct subproblem solution from a subproblem object.

Expects that the subproblem argument is a solved subproblem (i.e. one to which
numerical optimization has been applied).

Args:
    spbm: the subproblem structure.

Returns:
    sol: subproblem solution. =#
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
end # function

"""
    ptr_solve(pbm[, hot])

Solve the optimal control problem using the penalized trust region (PTR)
method.

# Arguments
- `pbm`: the trajectory problem to be solved.
- `warm`: (optional) warm start solution.

# Returns
- `sol`: the PTR solution structur.
- `history`: PTR iteration data history.
"""
function solve(pbm::SCPProblem,
               warm::Union{Nothing, SCPSolution}=nothing)::Tuple{
                   Union{SCPSolution, Nothing},
                   SCPHistory}
    # ..:: Initialize ::..

    if isnothing(warm)
        ref = generate_initial_guess(pbm)
    else
        ref = warm_start(pbm, warm)
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
        if k>pbm.pars.iter_max
            break
        end
    end

    reset(pbm.common.table)

    # ..:: Save solution ::..
    sol = SCPSolution(history)

    return sol, history
end # function

#= Compute the initial trajectory guess.

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to an SubproblemSolution structure.

Args:
    pbm: the PTR problem structure.

Returns:
    guess: the initial guess. =#
function generate_initial_guess(
    pbm::SCPProblem)::SubproblemSolution

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    guess = SubproblemSolution(x, u, p, 0, pbm)

    return guess
end # function

"""
    warm_start(pbm, warm)

Create initial guess from a warm start solution.

# Arguments
- `pbm`: the PTR problem structure.
- `warm`: warm start solution.

# Returns
- `guess`: the initial guess for PTR.
"""
function warm_start(pbm::SCPProblem,
                          warm::SCPSolution)::SubproblemSolution

    # Extract the warm-start trajectory
    x, u, p = warm.xd, warm.ud, warm.p
    guess = SubproblemSolution(x, u, p, 0, pbm)

    return guess
end # function

#= Add trust region constraint to the subproblem.

Args:
    spbm: the subproblem definition. =#
function add_trust_region!(spbm::Subproblem)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    q = spbm.def.pars.q_tr
    scale = spbm.def.common.scale
    traj = spbm.def.traj
    prg = spbm.prg
    x = spbm.x
    u = spbm.u
    p = spbm.p
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    uh_ref = scale.iSu*(spbm.ref.ud.-scale.cu)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)
    ηx = spbm.ηx
    ηu = spbm.ηu
    ηp = spbm.ηp

    q2cone = Dict(1 => L1, 2 => SOC, 4 => SOC, Inf => LINF)
    cone = q2cone[q]

    # >> Parameter trust region <<
    dp_lq = @new_variable(prg, "dp_lq")

    @add_constraint(prg, cone, "parameter_trust_region",
                    (p, dp_lq), begin # Value
                        local p, dp_lq = arg #noerr
                        local ph = scale.iSp*(p-scale.cp)
                        local dp = ph-ph_ref
                        vcat(dp_lq, dp)
                    end, begin # Jacobian
                        if LangServer; local J = Dict(); end
                        J[1] = vcat(zeros(traj.np)', scale.iSp)
                        J[2] = vcat(1, zeros(traj.np))
                    end)

    if q==4
        wp = @new_variable(prg, "wp")
        @add_constraint(prg, SOC, "parameter_trust_region",
                        (wp, dp_lq), begin # Value
                            local wp, dp_lq = arg #noerr
                            vcat(wp, dp_lq)
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = [1; 0]
                            J[2] = [0; 1]
                        end)
        @add_constraint(prg, GEOM, "parameter_trust_region",
                        (wp, ηp), begin # Value
                            local wp, ηp = arg #noerr
                            vcat(wp, ηp, 1)
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = [1; 0; 0]
                            J[2] = [0; 1; 0]
                        end)
    else
        @add_constraint(prg, NONPOS, "parameter_trust_region",
                        (ηp, dp_lq), begin # Value
                            local ηp, dp_lq = arg #noerr
                            dp_lq-ηp
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = [-1]
                            J[2] = [1]
                        end)
    end

    # State trust regions
    dx_lq = @new_variable(prg, N, "dx_lq")

    for k = 1:N
        @add_constraint(prg, cone, "state_trust_region",
                        (x[:, k], dx_lq[k]), begin # Value
                            local xk, dxk_lq = arg #noerr
                            local xhk = scale.iSx*(xk-scale.cx)
                            local dxk = xhk-xh_ref[:, k]
                            vcat(dxk_lq, dxk)
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = vcat(zeros(traj.nx)', scale.iSx)
                            J[2] = vcat(1, zeros(traj.nx))
                        end)
        if q==4
            # State
            wx = @new_variable(prg, "wx")
            @add_constraint(prg, SOC, "state_trust_region",
                            (wx, dx_lq[k]), begin # Value
                                local wx, dxk_lq = arg #noerr
                                vcat(wx, dxk_lq)
                            end, begin # Jacobian
                                if LangServer; local J = Dict(); end
                                J[1] = vcat(1, zeros(traj.nx))
                                J[2] = vcat(zeros(traj.nx)', I(traj.nx))
                            end)
            @add_constraint(prg, GEOM, "state_trust_region",
                            (wx, ηx[k]), begin # Value
                                local wx, ηxk = arg #noerr
                                vcat(wx, ηxk, 1)
                            end, begin # Jacobian
                                if LangServer; local J = Dict(); end
                                J[1] = [1; 0; 0]
                                J[2] = [0; 1; 0]
                            end)
        else
            # State
            @add_constraint(prg, NONPOS, "state_trust_region",
                            (ηx[k], dx_lq[k]), begin # Value
                                local ηxk, dxk_lq = arg #noerr
                                dxk_lq-ηxk
                            end, begin # Jacobian
                                if LangServer; local J = Dict(); end
                                J[1] = [-1]
                                J[2] = [1]
                            end)
        end
    end

    # Input trust regions
    du_lq = @new_variable(prg, N, "du_lq")

    for k = 1:N
        @add_constraint(prg, cone, "input_trust_region",
                        (u[:, k], du_lq[k]), begin # Value
                            local uk, duk_lq = arg #noerr
                            local uhk = scale.iSu*(uk-scale.cu)
                            local duk = uhk-uh_ref[:, k]
                            vcat(duk_lq, duk)
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = vcat(zeros(traj.nu)', scale.iSu)
                            J[2] = vcat(1, zeros(traj.nu))
                        end)
        if q==4
            wu = @new_variable(prg, "wu")
            @add_constraint(prg, SOC, "input_trust_region",
                            (wu, du_lq[k]), begin # Value
                                local wu, duk_lq = arg #noerr
                                vcat(wu, duk_lq)
                            end, begin # Jacobian
                                if LangServer; local J = Dict(); end
                                J[1] = vcat(1, zeros(traj.nu))
                                J[2] = vcat(zeros(traj.nu)', I(traj.nu))
                            end)
            @add_constraint(prg, GEOM, "input_trust_region",
                            (wu, ηu[k]), begin # Value
                                local wu, ηuk = arg #noerr
                                vcat(wu, ηuk, 1)
                            end, begin # Jacobian
                                if LangServer; local J = Dict(); end
                                J[1] = [1; 0; 0]
                                J[2] = [0; 1; 0]
                            end)
        else
            @add_constraint(prg, NONPOS, "input_trust_region",
                            (ηu[k], du_lq[k]), begin # Value
                                local ηuk, duk_lq = arg #noerr
                                duk_lq-ηuk
                            end, begin # Jacobian
                                if LangServer; local J = Dict(); end
                                J[1] = [-1]
                                J[2] = [1]
                            end)
        end
    end

    return nothing
end # function

#= Define the subproblem cost function.

Args:
    spbm: the subproblem definition. =#
function add_cost!(spbm::Subproblem)::Nothing

    # Compute the cost components
    compute_original_cost!(spbm)
    compute_trust_region_penalty!(spbm)
    compute_virtual_control_penalty!(spbm)

    spbm.J_aug = cost(spbm.prg)

    return nothing
end # function

"""
    original_cost(x, u, p, pbm)

Compute the original problem cost function.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `cost`: the original cost.
"""
function compute_original_cost!(spbm::Subproblem)::Nothing

    # Variables and parameters
    scp_iter = spbm.iter
    pbm = spbm.def
    N = pbm.pars.N
    t = pbm.common.t_grid
    traj_pbm = pbm.traj
    prg = spbm.prg
    x = spbm.x
    u = spbm.u
    p = spbm.p
    q = spbm.q

    x_stages = [x[:, k] for k=1:N]
    u_stages = [u[:, k] for k=1:N]

    spbm.J = @add_cost(
        prg, (x_stages..., u_stages..., p), (q,),
        begin
            local x = arg[1:N] #noerr
            local u = arg[(1:N).+N] #noerr
            local p, q = arg[end-1:end] #noerr

            # Terminal cost
            local xf = x[end]
            local J_term = isnothing(traj_pbm.φ) ? 0.0 :
                traj_pbm.φ(xf, p)

            # Integrated running cost
            local J_run = Vector{Objective}(undef, N)
            local ∇J_run = Vector{Dict}(undef, N)
            if !isnothing(traj_pbm.Γ)
                for k = 1:N
                    local out = traj_pbm.Γ(t[k], k, x[k], u[k], p, q)
                    if out isa Tuple
                        J_run[k], ∇J_run[k] = out
                    else
                        J_run[k] = out
                    end
                end
            else
                J_run[:] .= 0.0
            end
            local integ_J_run = trapz(J_run, t)

            J_term+integ_J_run
        end, begin # Jacobian
            local ∇g = ∇trapz(t)
            local idcs_x = 1:N
            local idcs_u = (1:N).+idcs_x[end]
            local idcs_p = idcs_u[end]+1
            local idcs_q = idcs_p[end]+1
            # Build Jacobian
            # Running cost
            for k = 1:N
                if isassigned(∇J_run, k)
                    zmap = Dict(:x=>idcs_x[k], :u=>idcs_u[k],
                                :p=>idcs_p, :q=>idcs_q)
                    for (key, val) in ∇J_run[k]
                        if key isa Symbol
                            key = zmap[key]
                        elseif key isa Tuple && length(key)==2
                            key = (zmap[key[1]], zmap[key[2]])
                        else
                            msg = @sprintf("Bad key (%s)", key)
                            err = SCPError(scp_iter, SCP_BAD_ARGUMENT, msg)
                            throw(err)
                        end
                        if haskey(J, key) #noerr
                            J[key] += val*∇g[k] #noerr
                        else
                            J[key] = val*∇g[k] #noerr
                        end
                    end
                end
            end
        end)

    return nothing
end # function

#= Compute the subproblem cost trust region penalty term.

Args:
    spbm: the subproblem definition. =#
function compute_trust_region_penalty!(spbm::Subproblem)::Nothing

    # Variables and parameters
    t = spbm.def.common.t_grid
    wtr = spbm.def.pars.wtr
    prg = spbm.prg
    ηx = spbm.ηx
    ηu = spbm.ηu
    ηp = spbm.ηp

    spbm.J_tr = @add_cost(
        prg, (ηx, ηu, ηp), begin # Value
            local ηx, ηu, ηp = arg #noerr
            ηp = ηp[1]
            wtr*(trapz(ηx, t)+trapz(ηu, t)+ηp)
        end, begin # Jacobian
            if LangServer; local J = Dict(); end
            J[1] = wtr*∇trapz(t)
            J[2] = wtr*∇trapz(t)
            J[3] = [wtr]
        end)

    return nothing
end # function

#= Compute the subproblem cost virtual control penalty term.

Args:
    spbm: the subproblem definition. =#
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

    # Compute virtual control penalty
    P = @new_variable(prg, N, "P")
    Pf = @new_variable(prg, 2, "Pf")
    if !isnothing(vs)
        for k = 1:N
            nvs = length(vs[:, k])
            if k<N
                sz_E_1 = size(E[:, :, k], 1)
                nvd = length(vd[:, k])
                @add_constraint(prg, L1, "vd_vs_penalty",
                                (P[k], vd[:, k], vs[:, k]), begin # Value
                                    local Pk, vdk, vsk = arg #noerr
                                    vcat(Pk, E[:, :, k]*vdk, vsk)
                                end, begin # Jacobian
                                    if LangServer; local J = Dict(); end
                                    J[1] = vcat(1, zeros(sz_E_1), zeros(nvs))
                                    J[2] = vcat(zeros(nvd)', E[:, :, k],
                                                zeros(nvs, nvd))
                                    J[3] = vcat(zeros(nvs)', zeros(sz_E_1, nvs),
                                                I(nvs))
                                end)
            else
                @add_constraint(prg, L1, "vd_vs_penalty",
                                (P[k], vs[:, k]), begin # Value
                                    local Pk, vsk = arg #noerr
                                    vcat(Pk, vsk)
                                end, begin # Jacobian
                                    if LangServer; local J = Dict(); end
                                    J[1] = vcat(1, zeros(nvs))
                                    J[2] = vcat(zeros(nvs)', I(nvs))
                                end)
            end
        end
    else
        for k = 1:N
            if k<N
                sz_E_1 = size(E[:, :, k], 1)
                nvd = length(vd[:, k])
                @add_constraint(prg, L1, "vd_vs_penalty",
                                (P[k], vd[:, k]), begin # Value
                                    local Pk, vdk = arg #noerr
                                    vcat(Pk, E[:, :, k]*vdk)
                                end, begin # Jacobian
                                    if LangServer; local J = Dict(); end
                                    J[1] = vcat(1, zeros(sz_E_1))
                                    J[2] = vcat(zeros(nvd)', E[:, :, k])
                                end)
            else
                @add_constraint(
                    prg, ZERO, "vd_vs_penalty",
                    (P[k],), begin # Value
                        local Pk, = arg #noerr
                        Pk
                    end, begin # Jacobian
                        if LangServer; local J = Dict(); end
                        J[1] = [1]
                    end)
            end
        end
    end

    if !isnothing(vic)
        nic = length(vic)
        @add_constraint(prg, L1, "vic_penalty",
                        (Pf[1], vic), begin # Value
                            local Pf1, vic = arg #noerr
                            vcat(Pf1, vic)
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = vcat(1, zeros(nic))
                            J[2] = vcat(zeros(nic)', I(nic))
                        end)
    else
        @add_constraint(prg, ZERO, "vic_penalty",
                        (Pf[1]), begin # Value
                            local Pf1, = arg #noerr
                            Pf1
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = [1]
                        end)
    end

    if !isnothing(vtc)
        ntc = length(vtc)
        @add_constraint(prg, L1, "vtc_penalty",
                        (Pf[2], vtc), begin # Value
                            local Pf2, vtc = arg #noerr
                            vcat(Pf2, vtc)
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = vcat(1, zeros(ntc))
                            J[2] = vcat(zeros(ntc)', I(ntc))
                        end)
    else
        @add_constraint(prg, ZERO, "vtc_penalty",
                        (Pf[2]), begin # Value
                            local Pf2, = arg #noerr
                            Pf2
                        end, begin # Jacobian
                            if LangServer; local J = Dict(); end
                            J[1] = [1]
                        end)
    end

    spbm.J_vc = @add_cost(
        prg, (P, Pf), begin # Value
            local P, Pf = arg #noerr
            wvc*(trapz(P, t)+sum(Pf))
        end, begin # Jacobian
            if LangServer; local J = Dict(); end
            J[1] = wvc*∇trapz(t)
            J[2] = wvc*ones(2)
        end)

    return nothing
end # function

#= Check if stopping criterion is triggered.

Args:
    spbm: the subproblem definition.

Returns:
    stop: true if stopping criterion holds. =#
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
    sol.improv_rel = (J_ref-J_sol)/abs(J_ref)

    # Compute stopping criterion
    stop = (spbm.iter>1 &&
        (sol.feas && (abs(sol.improv_rel)<=ε_rel || sol.deviation<=ε_abs)))

    return stop
end # function

#= Print command line info message.

Args:
    spbm: the subproblem that was solved.
    err: an PTR-specific error message. =#
function print_info(spbm::Subproblem,
                    err::Union{Nothing, SCPError}=nothing)::Nothing

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
        xh = scale.iSx*(sol.xd.-scale.cx)
        uh = scale.iSu*(sol.ud.-scale.cu)
        ph = scale.iSp*(sol.p-scale.cp)
        xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
        uh_ref = scale.iSu*(spbm.ref.ud.-scale.cu)
        ph_ref = scale.iSp*(spbm.ref.p-scale.cp)
        max_dxh = norm(xh-xh_ref, Inf)
        max_duh = norm(uh-uh_ref, Inf)
        max_dph = norm(ph-ph_ref, Inf)
        status = @sprintf "%s" sol.status
        status = status[1:min(8, length(status))]
        ΔJ = improvement_percent(sol.J_aug, ref.J_aug)
        ηx_max = maximum(sol.ηx)
        ηu_max = maximum(sol.ηu)
        ηp = sol.ηp

        # Associate values with columns
        assoc = Dict(:iter => spbm.iter,
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
                     :trp => ηp)

        # Set user-defined columns
        for col in traj.table_cols
            id, col_value = col[1], col[end]
            assoc[id] = col_value(sol.bay)
        end

        print(assoc, table)
    end

    overhead!(spbm)

    return nothing
end # function
