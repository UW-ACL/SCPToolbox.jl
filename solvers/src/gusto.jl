#= GuSTO algorithm data structures and methods.

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

using LinearAlgebra
using JuMP
using Printf

using Utils
using Parser

import ..ST, ..RealTypes, ..IntRange, ..RealVector, ..RealMatrix, ..Trajectory,
    ..Objective, ..VarArgBlk, ..CstArgBlk, ..DLTV

import ..SCPParameters, ..SCPSubproblem, ..SCPSubproblemSolution, ..SCPProblem,
    ..SCPSolution, ..SCPHistory

import ..warm_start
import ..discretize!
import ..add_dynamics!, ..add_convex_state_constraints!, ..add_convex_input_constraints!,
    ..add_nonconvex_constraints!, ..add_bcs!, ..correct_convex!
import ..solve_subproblem!, ..solution_deviation, ..unsafe_solution,
    ..overhead!, ..save!, ..get_time

const CLP = ConicLinearProgram
const Variable = ST.Variable
const Optional = ST.Optional
const Func = ST.Func
const OptVarArgBlk = Optional{VarArgBlk}

export Parameters

""" Structure holding the GuSTO algorithm parameters."""
struct Parameters <: SCPParameters
    N::Int               # Number of temporal grid nodes
    Nsub::Int            # Number of subinterval integration time nodes
    iter_max::Int        # Maximum number of iterations
    disc_method::DiscretizationType # The discretization method
    λ_init::RealTypes    # Initial soft penalty weight
    λ_max::RealTypes     # Maximum soft penalty weight
    ρ_0::RealTypes       # Trust region update threshold (lower, good solution)
    ρ_1::RealTypes       # Trust region update threshold (upper, bad solution)
    β_sh::RealTypes      # Trust region shrinkage factor
    β_gr::RealTypes      # Trust region growth factor
    γ_fail::RealTypes    # Soft penalty weight growth factor
    η_init::RealTypes    # Initial trust region radius
    η_lb::RealTypes      # Minimum trust region radius
    η_ub::RealTypes      # Maximum trust region radius
    μ::RealTypes         # Exponential shrink rate for trust region
    iter_μ::RealTypes    # Iteration at which to apply trust region shrink
    ε_abs::RealTypes     # Absolute convergence tolerance
    ε_rel::RealTypes     # Relative convergence tolerance
    feas_tol::RealTypes  # Dynamic feasibility tolerance
    pen::Symbol          # Penalty type (:quad, :softplus)
    hom::RealTypes       # Homotopy parameter to use when pen==:softplus
    q_tr::RealTypes      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::RealTypes    # Stopping criterion norm
    solver::Module       # The numerical solver to use for the subproblems
    solver_opts::Dict{String, Any} # Numerical solver options
end

""" GuSTO subproblem solution."""
mutable struct SubproblemSolution <: SCPSubproblemSolution
    iter::Int             # GuSTO iteration number
    # >> Discrete-time rajectory <<
    xd::RealMatrix        # States
    ud::RealMatrix        # Inputs
    p::RealVector         # Parameter vector
    # >> Cost values <<
    J::RealTypes          # The original cost
    J_st::RealTypes       # The state constraint soft penalty
    J_tr::RealTypes       # The trust region soft penalty
    J_aug::RealTypes      # Overall nonlinear cost
    L::RealTypes          # J *linearized* about reference solution
    L_st::RealTypes       # J_st *linearized* about reference solution
    L_aug::RealTypes      # Overall convex cost
    # >> Trajectory properties <<
    status::ST.ExitStatus # Numerical optimizer exit status
    feas::Bool            # Dynamic feasibility flag
    defect::RealMatrix    # "Defect" linearization accuracy metric
    deviation::RealTypes  # Deviation from reference trajectory
    unsafe::Bool          # Indicator that the solution is unsafe to use
    cost_error::RealTypes # Cost error committed
    dyn_error::RealTypes  # Cumulative dynamics error committed
    ρ::RealTypes          # Convexification performance metric
    tr_update::String     # Growth direction indicator for trust region
    λ_update::String      # Growth direction indicator for soft penalty weight
    reject::Bool          # Indicator whether GuSTO rejected this solution
    dyn::DLTV             # The dynamics
    bay::Dict             # Storage bay for user-set values during callback
end

""" Subproblem definition for the convex numerical optimizer. """
mutable struct Subproblem <: SCPSubproblem
    iter::Int                  # GuSTO iteration number
    prg::ConicProgram          # The optimization problem handle
    algo::String               # SCP and convex algorithms used
    # >> Algorithm parameters <<
    def::SCPProblem            # The GuSTO algorithm definition
    λ::RealTypes               # Soft penalty weight
    η::RealTypes               # Trust region radius
    κ::RealTypes               # Trust region shrinking multiplier
    # >> Reference and solution trajectories <<
    sol::Union{SubproblemSolution, Missing} # Solution trajectory
    ref::Union{SubproblemSolution, Missing} # Reference trajectory
    # >> Cost function <<
    L::Objective       # The original cost
    L_st::Objective    # The state constraint soft penalty
    L_tr::Objective    # The trust region soft penalty
    L_aug::Objective   # Overall cost
    # >> Physical variables <<
    x::VarArgBlk       # Discrete-time states
    u::VarArgBlk       # Discrete-time inputs
    p::VarArgBlk       # Parameters
    # >> Statistics <<
    timing::Dict{Symbol, RealTypes} # Runtime profiling
end

"""
    create(pars, traj)

Construct the GuSTO problem definition.

# Arguments
- `pars`: GuSTO algorithm parameters.
- `traj`: the underlying trajectory optimization problem.

# Returns
- `pbm`: the problem structure ready for being solved by GuSTO.
"""
function create(
        pars::Parameters,
        traj::TrajectoryProblem
)::SCPProblem

    default_columns = [
        # Iteration count
        (:iter, "k", "%d", 2),
        # Solver status
        (:status, "status", "%s", 8),
        # Overall cost (including penalties)
        (:cost, "J", "%.2e", 9),
        # Maximum deviation in state
        (:dx, "Δx", "%.0e", 5),
        # Maximum deviation in input
        (:du, "Δu", "%.0e", 5),
        # Maximum deviation in input
        (:dp, "Δp", "%.0e", 5),
        # Dynamic feasibility flag (true or false)
        (:dynfeas, "dyn", "%s", 3),
        # Trust region size
        (:tr, "η", "%.2f", 5),
        # Soft penalty weight
        (:soft, "λ", "%.0e", 5),
        # Convexification performance metric
        (:ρ, "ρ", "%s", 9),
        # Cost improvement (percent)
        (:cost_improv, "ΔJ %", "%s", 9),
        # Update direction for trust region radius (grow? shrink?)
        (:dtr, "Δη", "%s", 3),
        # Update direction for soft penalty weight (grow? shrink?)
        (:dλ, "Δλ", "%s", 3),
        # Reject solution indicator
        (:rej, "rej", "%s", 5)
    ]

    # User-defined extra columns
    user_columns = [tuple(col[1:4]...) for col in traj.table_cols]

    all_columns = [default_columns; user_columns]

    table = ST.Table(all_columns)

    pbm = SCPProblem(pars, traj, table)

    return pbm
end

"""
    Subproblem(pbm[, iter, λ, η, ref])

Constructor for an empty convex optimization subproblem. No cost or
constraints. Just the decision variables and empty associated parameters.

# Arguments
- `pbm`: the GuSTO problem being solved.
- `iter`: (optional) GuSTO iteration number.
- `λ`: (optional) the soft penalty weight.
- `η`: (optional) the trust region radius.
- `ref`: (optional) the reference trajectory.

# Returns
- `spbm`: the subproblem structure.
"""
function Subproblem(
        pbm::SCPProblem,
        iter::Int=0,
        λ::RealTypes=1e4,
        η::RealTypes=1.0,
        ref::Union{SubproblemSolution, Missing}=missing
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

    # Optimization problem handle
    solver = pars.solver
    solver_opts = pars.solver_opts
    prg = ConicProgram(
        pbm.traj;
        solver=solver.Optimizer,
        solver_options=solver_opts
    )
    cvx_algo = string(pars.solver)
    algo = @sprintf("GuSTO (backend: %s)", cvx_algo)

    sol = missing # No solution associated yet with the subproblem

    # Cost function
    L = missing
    L_st = missing
    L_tr = missing
    L_aug = missing

    # Physical decision variables
    x = @new_variable(prg, (nx, N), "x")
    u = @new_variable(prg, (nu, N), "u")
    p = @new_variable(prg, np, "p")
    Sx = diag(scale.Sx)
    Su = diag(scale.Su)
    Sp = diag(scale.Sp)
    @scale(x, Sx, scale.cx)
    @scale(u, Su, scale.cu)
    @scale(p, Sp, scale.cp)

    # Trust region shrink factor
    κ = (iter < pars.iter_μ) ? 1.0 : pars.μ^(1+iter-pars.iter_μ)

    spbm = Subproblem(
        iter, prg, algo, pbm, λ, η, κ, sol, ref, L, L_st,
        L_tr, L_aug, x, u, p, timing)

    return spbm
end

"""
    SubproblemSolution(x, u, p, iter, pbm)

Construct a subproblem solution from a discrete-time trajectory. This leaves
parameters of the solution other than the passed discrete-time trajectory
unset.

# Arguments
- `x`: discrete-time state trajectory.
- `u`: discrete-time input trajectory.
- `p`: parameter vector.
- `iter`: GuSTO iteration number.
- `pbm`: the GuSTO problem definition.

# Returns
- `subsol`: subproblem solution structure.
"""
function SubproblemSolution(
        x::RealMatrix,
        u::RealMatrix,
        p::RealVector,
        iter::Int,
        pbm::SCPProblem
)::SubproblemSolution

    # Parameters
    N = pbm.pars.N
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    nv = size(pbm.common.E, 2)
    disc = pbm.pars.disc_method

    # Uninitialized parts
    status = MOI.OPTIMIZE_NOT_CALLED
    feas = false
    defect = fill(NaN, nx, N-1)
    deviation = NaN
    unsafe = false
    cost_error = NaN
    dyn_error = NaN
    ρ = NaN
    tr_update = ""
    λ_update = ""
    reject = false
    dyn = DLTV(nx, nu, np, nv, N, disc)
    bay = Dict()

    J = NaN
    J_st = NaN
    J_tr = NaN
    J_aug = NaN
    L = NaN
    L_st = NaN
    L_aug = NaN

    subsol = SubproblemSolution(
        iter, x, u, p, J, J_st, J_tr, J_aug, L, L_st, L_aug, status, feas,
        defect, deviation, unsafe, cost_error, dyn_error, ρ, tr_update,
        λ_update, reject, dyn, bay)

    # Compute the DLTV dynamics around this solution
    discretize!(subsol, pbm)

    return subsol
end

"""
    SubproblemSolution(spbms)

Construct subproblem solution from a subproblem object. Expects that the
subproblem argument is a solved subproblem (i.e. one to which numerical
optimization has been applied).

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

    # Save the optimal cost values
    sol.J = original_cost(x, u, p, spbm, :nonconvex)
    sol.J_st = state_penalty_cost(x, p, spbm, :nonconvex)
    sol.J_tr = value(spbm.L_tr)
    sol.J_aug = sol.J+sol.J_st+sol.J_tr
    sol.L = value(spbm.L)
    sol.L_st = value(spbm.L_st)
    sol.L_aug = value(spbm.L_aug)

    return sol
end

"""
    gusto_solve(pbm)

Apply the GuSTO algorithm to solve the trajectory generation problem.

# Arguments
- `pbm`: the trajectory problem to be solved.

# Returns
- `sol`: the GuSTO solution structure.
- `history`: GuSTO iteration data history.
"""
function solve(
        pbm::SCPProblem,
        warm::Union{Nothing, SCPSolution}=nothing
)::Tuple{Union{SCPSolution, SubproblemSolution, Nothing}, SCPHistory}

    # ..:: Initialize ::..

    λ = pbm.pars.λ_init
    η = pbm.pars.η_init
    if isnothing(warm)
        ref = generate_initial_guess(pbm)
    else
        ref = warm_start(pbm, warm)
    end

    history = SCPHistory()

    callback_fun! = pbm.traj.callback!
    user_callback = !isnothing(callback_fun!)

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        spbm = Subproblem(pbm, k, λ, η, ref)

        add_cost!(spbm)
        add_dynamics!(spbm; relaxed=false)
        add_convex_input_constraints!(spbm)
        add_bcs!(spbm; relaxed=false)

        save!(history, spbm)

        try
            # >> Solve the subproblem <<
            solve_subproblem!(spbm)

            # "Emergency exit" the GuSTO loop if something bad happened
            # (e.g. numerical problems)
            if unsafe_solution(spbm)
                print_info(spbm)
                break
            end

            # >> Check stopping criterion <<
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

            # Update trust region
            ref, η, λ = update_trust_region!(spbm)
        catch e
            isa(e, SCPError) || rethrow(e)
            print_info(spbm, e)
            break
        end

        # >> Print iteration info <<
        print_info(spbm)
    end

    reset(pbm.common.table)

    # ..:: Save solution ::..
    sol = SCPSolution(history)

    return sol, history
end

"""
    generate_initial_guess(pbm)

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to a SubproblemSolution structure.

# Arguments
- `pbm`: the GuSTO problem structure.

# Returns
- `guess`: the initial guess.
"""
function generate_initial_guess(pbm::SCPProblem)::SubproblemSolution

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    correct_convex!(x, u, p, pbm, (pbm) -> Subproblem(pbm))
    guess = SubproblemSolution(x, u, p, 0, pbm)

    return guess
end

"""
    add_cost!(spbm)

Define the subproblem cost function.

# Arguments
- `spbm`: the subproblem definition.
"""
function add_cost!(spbm::Subproblem)::Nothing

    # Variables and parameters
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Compute the cost components
    spbm.L = original_cost(x, u, p, spbm)
    spbm.L_st = state_penalty_cost(x, p, spbm)
    spbm.L_tr = trust_region_cost(x, p, spbm)

    # Overall cost
    spbm.L_aug = cost(spbm.prg)

    return nothing
end

"""
    original_cost(x, u, p, spbm[, mode])

Compute the original cost function. This function has two "modes": the
(default) convex mode computes the convex version of the cost (where all
non-convexity has been convexified), while the nonconvex mode computes the
fully nonlinear cost.

# Arguments
- `x`: the discrete-time state trajectory.
- `u`: the discrete-time input trajectory.
- `p`: the parameter vector.
- `spbm`: the subproblem structure.
- `mode`: (optional) either :convex (default) or :nonconvex.

# Returns
- `cost`: the original cost.
"""
function original_cost(
        x::Union{VarArgBlk, RealMatrix},
        u::Union{VarArgBlk, RealMatrix},
        p::Union{VarArgBlk, RealVector},
        spbm::Subproblem,
        mode::Symbol=:convex
)::Objective

    # Parameters
    pars = spbm.def.pars
    traj = spbm.def.traj
    prg = spbm.prg
    N = pars.N
    t = spbm.def.common.t_grid
    no_running_cost = isnothing(traj.S) && isnothing(traj.ℓ) && isnothing(traj.g)

    if mode==:convex

        # Reference trajectory
        ref = spbm.ref
        xb = ref.xd
        ub = ref.ud
        pb = ref.p

        x_stages = [x[:, k] for k=1:N]
        u_stages = [u[:, k] for k=1:N]

        cost = @add_cost(
            prg, (x_stages..., u_stages..., p),
            begin
                local x = arg[1:N]
                local u = arg[(1:N).+N]
                local p = arg[end]

                # Terminal cost
                local xf = x[end]
                local J_term = isnothing(traj.φ) ? 0.0 : traj.φ(xf, p)

                # Integrated running cost
                local J_run = Vector{Objective}(undef, N)
                for k = 1:N
                    tkp = (t[k], k, p)
                    tkxp = (t[k], k, x[k], p)
                    if no_running_cost
                        J_run[k] = 0.0
                    else
                        tkpb = (t[k], k, pb)
                        tkxpb = (t[k], k, xb[:, k], pb)
                        Γk = 0.0
                        if !isnothing(traj.S)
                            if traj.S_cvx
                                Γk += u[k]'*traj.S(tkp...)*u[k]
                            else
                                uSu = ub[:, k]'*traj.S(tkpb...)*ub[:, k]
                                ∇u_uSu = 2*traj.S(tkpb...)*ub[:, k]
                                ∇p_S = traj.dSdp(pb)
                                ∇p_uSu = [ub[:, k]'*∇p_S[i]*ub[:, k] for i=1:traj.np]
                                du = u[k]-ub[:, k]
                                dp = p-pb
                                uSu1 = uSu+∇u_uSu'*du+∇p_uSu.*dp
                                Γk += uSu1
                            end
                        end
                        if !isnothing(traj.ℓ)
                            if traj.ℓ_cvx
                                Γk += u[k]'*traj.ℓ(tkxp...)
                            else
                                uℓ = ub[:, k]'*traj.ℓ(tkxpb...)
                                ∇u_uℓ = traj.ℓ(tkxpb...)
                                ∇x_uℓ = !isnothing(traj.dℓdx) ? traj.dℓdx(tkxpb...)'*ub[:, k] :
                                    zeros(traj.nx)
                                ∇p_uℓ = !isnothing(traj.dℓdp) ? traj.dℓdp(tkxpb...)'*ub[:, k] :
                                    zeros(traj.np)
                                du = u[k]-ub[:, k]
                                dx = x[k]-xb[:, k]
                                dp = p-pb
                                uℓ1 = uℓ+∇u_uℓ'*du+∇x_uℓ'*dx+∇p_uℓ'*dp
                                Γk += uℓ1
                            end
                        end
                        if !isnothing(traj.g)
                            if traj.g_cvx
                                Γk += traj.g(tkxp...)
                            else
                                g = traj.g(tkxpb...)
                                ∇x_g = !isnothing(traj.dgdx) ? traj.dgdx(tkxpb...) : zeros(traj.nx)
                                ∇p_g = !isnothing(traj.dgdp) ? traj.dgdp(tkxpb...) : zeros(traj.np)
                                dx = x[k]-xb[:, k]
                                dp = p-pb
                                g1 = g+∇x_g'*dx+∇p_g'*dp
                                Γk += g1
                            end
                        end
                        J_run[k] = Γk
                    end
                end
                local integ_J_run = trapz(J_run, t)

                J_term+integ_J_run
            end)

    else

        # Terminal cost
        xf = x[:, end]
        J_term = isnothing(traj.φ) ? 0.0 : traj.φ(xf, p)

        # Integrated running cost
        J_run = Vector{Objective}(undef, N)
        for k = 1:N
            tkp = (t[k], k, p)
            tkxp = (t[k], k, x[:, k], p)
            if no_running_cost
                cost_run_integrand[k] = 0.0
            else
                Γk = 0.0
                Γk += !isnothing(traj.S) ? u[:, k]'*traj.S(tkp...)*u[:, k] : 0.0
                Γk += !isnothing(traj.ℓ) ? u[:, k]'*traj.ℓ(tkxp...) : 0.0
                Γk += !isnothing(traj.g) ? traj.g(tkxp...) : 0.0
                J_run[k] = Γk
            end
        end
        integ_J_run = trapz(J_run, t)

        cost = J_term+integ_J_run
    end

    return cost
end

"""
    state_penalty_cost(x, p, spbm[, mode])

Compute a soft penalty cost on the state constraints. This includes the convex
state constraints x∈X and the generally nonconvex path constraints s(x, u,
p)<=0.

# Arguments
- `x`: the discrete-time state trajectory.
- `p`: the parameter vector.
- `spbm`: the subproblem structure.
- `mode`: (optional) either :convex (default) or :nonconvex.

# Returns
- `cost_st`: the original cost.
"""
function state_penalty_cost(
        x::Union{VarArgBlk, RealMatrix},
        p::Union{VarArgBlk, RealVector},
        spbm::Subproblem,
        mode::Symbol=:convex
)::Objective

    # Parameters
    pbm = spbm.def
    pars = pbm.pars
    traj = pbm.traj
    prg = spbm.prg
    N = pars.N
    t = pbm.common.t_grid
    nx = traj.nx
    np = traj.np

    if mode==:convex

        # Reference trajectory
        ref = spbm.ref
        xb = ref.xd
        pb = ref.p

        # Convex state constraints
        # Flatten them with a map that relates the temporal index k to the flattened indices
        cost_cvx_all = convex_state_penalty(x, p, spbm)
        cost_cvx_var, cost_cvx_pen, cvx_k2j, j = [], [], Dict{Int, Vector{Int}}(), 1
        for k = 1:length(cost_cvx_all)
            cvx_k2j[k] = Int[]
            for i = 1:length(cost_cvx_all[k])
                push!(cost_cvx_var, cost_cvx_all[k][i][1])
                push!(cost_cvx_pen, cost_cvx_all[k][i][2])
                push!(cvx_k2j[k], j)
                j += 1
            end
        end

        # Nonconvex path constraints
        cost_ncvx_all = []
        if !isnothing(traj.s)
            for k = 1:N
                pen_k = []
                tkxp = (t[k], k, xb[:, k], pb)
                s = traj.s(tkxp...)
                dsdx = !isnothing(traj.C) ? traj.C(tkxp...) : zeros(length(s), nx)
                dsdp = !isnothing(traj.G) ? traj.G(tkxp...) : zeros(length(s), np)
                for i = 1:length(s)
                    push!(pen_k, soft_penalty(
                        spbm, s[i], dsdx[i, :], dsdp[i, :], x[:, k], p, xb[:, k], pb))
                end
                push!(cost_ncvx_all, pen_k)
            end
        end
        # Flatten the nonconvex constraints, with a map that relates the temporal index k to the
        # flattened indices
        cost_ncvx_var, cost_ncvx_pen, ncvx_k2j, j = [], [], Dict{Int, Vector{Int}}(), 1
        for k = 1:length(cost_ncvx_all)
            ncvx_k2j[k] = Int[]
            for i = 1:length(cost_ncvx_all[k])
                push!(cost_ncvx_var, cost_ncvx_all[k][i][1])
                push!(cost_ncvx_pen, cost_ncvx_all[k][i][2])
                push!(ncvx_k2j[k], j)
                j += 1
            end
        end

        cost_st = @add_cost(
            prg, (cost_cvx_var..., cost_ncvx_var...),
            begin
                local var_cvx = arg[1:length(cost_cvx_var)]
                local var_ncvx = arg[(length(cost_cvx_var)+1):end]

                local cost_st = 0.0 # The penalty cost

                # Penalty cost for convex state constraints
                if length(var_cvx)>0
                    local cost_soft_cvx = []
                    for k in keys(cvx_k2j)
                        push!(cost_soft_cvx, 0.0)
                        for j in cvx_k2j[k]
                            cost_soft_cvx[end] += cost_cvx_pen[j](var_cvx[j])
                        end
                    end
                    cost_st += trapz(cost_soft_cvx, t)
                end

                # Penalty cost for nonconvex path constraints
                if length(var_ncvx)>0
                    local cost_soft_ncvx = []
                    for k in keys(ncvx_k2j)
                        push!(cost_soft_ncvx, 0.0)
                        for j in ncvx_k2j[k]
                            cost_soft_ncvx[end] += cost_ncvx_pen[j](var_ncvx[j])
                        end
                    end
                    cost_st += trapz(cost_soft_ncvx, t)
                end

                cost_st
            end)

    else

        # Convex state constraints
        cost_st_all_time_nodes = convex_state_penalty(x, p, spbm)
        if length(cost_st_all_time_nodes)>0
            cost_st = trapz([sum(cost_st_k, init=0) for cost_st_k in cost_st_all_time_nodes], t)
        else
            cost_st = 0.0
        end

        # Nonconvex path constraints
        if !isnothing(traj.s)
            cost_soft_s = Vector{Objective}(undef, N)
            for k = 1:N
                cost_soft_s[k] = 0.0
                s = traj.s(t[k], k, x[:, k], p)
                for i = 1:length(s)
                    cost_soft_s[k] += soft_penalty(spbm, s[i])
                end
            end
            cost_st += trapz(cost_soft_s, t)
        end

    end

    return cost_st
end

"""
    convex_state_penalty(x, p, spbm)

Compute a penalty for convex state path constraints.

# Arguments
- `x`: the discrete-time state trajectory.
- `p`: the parameter vector.
- `spbm`: the subproblem structure.

# Returns
- `pen`: list of penalties per-time-step, where the element for each time step is a list of
  penalties per (scalar) state constraint.
"""
function convex_state_penalty(
        x::Union{VarArgBlk, RealMatrix},
        p::Union{VarArgBlk, RealVector},
        spbm::Subproblem,
)::Vector

    # Parameters
    pbm = spbm.def
    pars = pbm.pars
    traj = pbm.traj
    N = pars.N
    t = pbm.common.t_grid

    pen = []
    if !isnothing(traj.X)
        for k = 1:N
            cone_indicators = traj.X(t[k], k, x[:, k], p)
            pen_k = []
            for ρ in cone_indicators
                for i=1:length(ρ)
                    push!(pen_k, soft_penalty(spbm, ρ[i]))
                end
            end
            push!(pen, pen_k)
        end
    end

    return pen
end

"""
    soft_penalty(spbm, f[, dfdx, dfdp, dx, dp])

Compute a smooth, convex and nondecreasing penalization function. Basic idea:
the penalty is zero if quantity f<0, else the penalty is positive (and grows as
f becomes more positive). If the Jacobian values are passed in, a linearized
version of the penalty function is computed. If a Jacobians (e.g. d/dx, or
d/dp) is zero, then you can pass `nothing` in its place.

# Arguments
- `spbm`: the subproblem structure.
- `f`: the quantity to be penalized.
- `dfdx`: (optional) Jacobian of f wrt state.
- `dfdp`: (optional) Jacobian of f wrt parameter vector.
- `dx`: (optional) state vector deviation from reference.
- `dp`: (optional) parameter vector deviation from reference.

# Returns
- Numerical mode: the penalization function value.
- Optimization mode: a (variable, function) tuple where the function transforms the variable into
  the penalization function value.
"""
function soft_penalty(
        spbm::Subproblem,
        f::Union{RealTypes, VarArgBlk},
        dfdx::Union{RealVector, Nothing}=nothing,
        dfdp::Union{RealVector, Nothing}=nothing,
        x::Union{VarArgBlk, Nothing}=nothing,
        p::Union{VarArgBlk, Nothing}=nothing,
        xb::Union{RealVector, Nothing}=nothing,
        pb::Union{RealVector, Nothing}=nothing
)::Union{RealTypes, Tuple{VarArgBlk, Func}}

    # Parameters
    pars = spbm.def.pars
    traj = spbm.def.traj
    prg = spbm.prg
    penalty = pars.pen
    hom = pars.hom
    λ = spbm.λ
    linearized = !isnothing(dfdx) || !isnothing(dfdp)
    numerical_mode = f isa RealTypes && (
        !linearized || (linearized && (x isa RealVector && p isa RealVector)))

    # Get linearized version of the quantity being penalized, if applicable
    if linearized
        if numerical_mode
            f = f+dfdx'*(x-xb)+dfdp'*(p-pb)
        else
            f_lin = (x, p) -> f+dfdx'*(x-xb)+dfdp'*(p-pb)
        end
    end

    # Compute the function value
    # The possibilities are:
    #   (:quad)      h(f(x, p)) = λ*(max(0, f(x, p)))^2
    #   (:softplus)  h(f(x, p)) = λ*log(1+exp(hom*f(x, p)))/hom
    if penalty==:quad
        # ..:: Quadratic penalty ::..
        if numerical_mode
            return λ*(max(0.0, f))^2
        else
            u = @new_variable(prg, "u")
            v = @new_variable(prg, "v")
            @add_constraint(
                prg, NONPOS, (u,),
                begin
                    local u, = arg
                    -u[1]
                end)
            if linearized
                @add_constraint(
                    prg, NONPOS, (x, p, u, v), begin
                        local x, p, u, v = arg
                        f_lin(x, p)+u[1]-v[1]
                    end)
            else
                @add_constraint(
                    prg, NONPOS, (f, u, v), begin
                        local f, u, v = arg
                        f[1]+u[1]-v[1]
                    end)
            end
            return v, (v) -> λ*v[1]^2
        end
    else
        # ..:: Log-sum-exp penalty ::..
        if numerical_mode
            F = [0, f]
            return λ*logsumexp(F; t=hom)
        else
            u = @new_variable(prg, "u")
            v = @new_variable(prg, "v")
            w = @new_variable(prg, "w")
            @add_constraint(
                prg, EXP, (w, u),
                begin
                    local w, u = arg
                    vcat(-w, 1, u)
                end)
            if linearized
                @add_constraint(
                    prg, EXP, (x, p, w, v), begin
                        local x, p, w, v = arg
                        vcat(hom*f_lin(x, p)-w[1], 1, v[1])
                    end)
            else
                @add_constraint(
                    prg, EXP, (f, w, v), begin
                        local f, w, v = arg
                        vcat(hom*f[1]-w[1], 1, v[1])
                    end)
            end
            @add_constraint(
                prg, NONPOS, (u, v),
                begin
                    local u, v = arg
                    u[1]+v[1]-1
                end)
            return w, (w) -> λ*w[1]/hom
        end
    end
end

"""
    trust_region_cost(x, p, spbm[, mode])

Compute the trust region constraint soft penalty. This function has two
"modes": the (default) convex mode computes the convex version of the cost
(where all non-convexity has been convexified), while the nonconvex mode
computes the fully nonlinear cost.

# Arguments
- `x`: the discrete-time state trajectory.
- `p`: the parameter vector.
- `spbm`: the subproblem structure.
- `mode`: (optional) either :convex (default) or :nonconvex. In the convex mode the function
  integrates and return the integrated penalty. In the nonconvex mode, it returns the actual
  trust region left-hand sides (which should be <=0 if the trust region constraints are
  satisfied).

# Returns
- `cost_tr`: if :convex then return the trust region soft penalty cost; or
- `tr`: if :nonconvex, return the trust regions left-hand sides (which should all be
  <=0 if the trust region constraints are satisfied).
"""
function trust_region_cost(
        x::Union{VarArgBlk, RealMatrix},
        p::Union{VarArgBlk, RealVector},
        spbm::Subproblem,
        mode::Symbol=:convex
)::Union{Objective, RealVector}

    # Parameters
    prg = spbm.prg
    pars = spbm.def.pars
    scale = spbm.def.common.scale
    q = pars.q_tr
    N = pars.N
    η = spbm.η
    t = spbm.def.common.t_grid

    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)

    q2cone = Dict(1 => L1, 2 => SOC, 4 => SOC, Inf => LINF)
    cone = q2cone[q]

    if mode==:convex

        tr = @new_variable(prg, N, "tr")
        dx_lq = @new_variable(prg, N, "dx_lq")
        dp_lq = @new_variable(prg, "dp_lq")

        # Parameter trust region
        @add_constraint(
            prg, cone, "parameter_trust_region", (p, dp_lq),
            begin
                local p, dp_lq = arg
                local ph = scale.iSp*(p-scale.cp)
                local dp = ph-ph_ref
                vcat(dp_lq, dp)
            end)

        for k = 1:N

            # State trust region
            @add_constraint(
                prg, cone, "state_trust_region", (x[:, k], dx_lq[k]),
                begin
                    local xk, dxk_lq = arg
                    local xhk = scale.iSx*(xk-scale.cx)
                    local dxk = xhk-xh_ref[:, k]
                    vcat(dxk_lq, dxk)
                end)

            # Trust region bound
            if q==4
                w = @new_variable(prg, "w")
                @add_constraint(
                    prg, SOC, "trust_region_bound", (dx_lq[k], dp_lq, w),
                    begin
                        local dxk_lq, dp_lq, w = arg
                        vcat(w, dxk_lq, dp_lq)
                    end)
                @add_constraint(
                    prg, GEOM, "trust_region_bound", (w, η, tr[k]),
                    begin
                        local w, η, trk = arg
                        vcat(w, η+trk[1], 1)
                    end)
            else
                @add_constraint(
                    prg, NONPOS, "trust_region_bound", (dx_lq[k], dp_lq, tr[k]),
                    begin
                        local dxk_lq, dp_lq, trk = arg
                        dxk_lq[1]+dp_lq[1]-(η+trk[1])
                    end)
            end

        end

        # Compute the soft penalty terms for the cost function
        pen_terms = [soft_penalty(spbm, tr[k]) for k=1:N]
        pen_var = [pen[1] for pen in pen_terms]
        pen_map = [pen[2] for pen in pen_terms]

        cost_tr = @add_cost(
            prg, (pen_var...,), begin
                local pen_var = arg[1:end]
                local cost_tr_integrand = []
                for k=1:N
                    push!(cost_tr_integrand, pen_map[k](pen_var[k]))
                end
                trapz(cost_tr_integrand, t)
            end)

        return cost_tr
    else

        tr = RealVector(undef, N)

        xh = scale.iSx*(x.-scale.cx)
        ph = scale.iSp*(p-scale.cp)
        dx = xh-xh_ref
        dp = ph-ph_ref

        dp_lq = norm(dp, q)
        for k = 1:N
            dx_lq = norm(dx[:, k], q)
            w = (q==4) ? 2 : 1
            tr[k] = dx_lq^w+dp_lq^w-η
        end

        return tr
    end

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
    λ = spbm.λ
    λ_max = pbm.pars.λ_max

    # Compute solution deviation from reference
    sol.deviation = solution_deviation(spbm)

    # Compute cost improvement
    ΔJ = abs(ref.J_aug-sol.J_aug)/abs(ref.J_aug)

    # Check infeasibility
    infeas = λ>λ_max

    # Compute stopping criterion
    stop = ((spbm.iter>1) &&
            ((sol.feas && (ΔJ<=ε_rel || sol.deviation<=ε_abs)) ||
             infeas))

    return stop
end

"""
    update_trust_region!(spbm)

Compute the new reference, trust region, and soft penalty.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `next_ref`: reference trajectory for the next iteration.
- `next_η`: trust region radius for the next iteration.
- `next_λ`: soft penalty weight for the next iteration.
"""
function update_trust_region!(spbm::Subproblem)::Tuple{
        SubproblemSolution,
        RealTypes,
        RealTypes}

    # Parameters
    pbm = spbm.def
    traj = pbm.traj
    N = pbm.pars.N
    t = pbm.common.t_grid
    sol = spbm.sol
    ref = spbm.ref
    xb = ref.xd
    ub = ref.ud
    pb = ref.p
    x = sol.xd
    u = sol.ud
    p = sol.p

    # Cost error
    J, L = sol.J_aug, sol.L_aug
    sol.cost_error = abs(J-L)
    cost_nrml = abs(L)

    # Dynamics error
    Δf = RealVector(undef, N)
    dxdt = RealVector(undef, N)
    for k = 1:N
        f = traj.f(t[k], k, xb[:, k], ub[:, k], pb)
        A = traj.A(t[k], k, xb[:, k], ub[:, k], pb)
        B = traj.B(t[k], k, xb[:, k], ub[:, k], pb)
        F = traj.F(t[k], k, xb[:, k], ub[:, k], pb)
        r = f-A*xb[:, k]-B*ub[:, k]-F*pb
        f_lin = A*x[:, k]+B*u[:, k]+F*p+r
        f_nl = traj.f(t[k], k, x[:, k], u[:, k], p)
        Δf[k] = norm(f_nl-f_lin)
        dxdt[k] = norm(f_lin)
    end
    sol.dyn_error = trapz(Δf, t)
    dynamics_nrml = trapz(dxdt, t)

    # Convexification performance metric
    normalization_term = cost_nrml+dynamics_nrml
    sol.ρ = (sol.cost_error+sol.dyn_error)/normalization_term

    # Apply update rule
    next_ref, next_η, next_λ = update_rule!(spbm)

    return next_ref, next_η, next_λ
end

"""
    update_rule!(spbm)

Apply the low-level GuSTO trust region update rule. Based on the obtained
subproblem solution, this computes the new trust region value, soft penalty
weight, and reference trajectory.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `next_ref`: reference trajectory for the next iteration.
- `next_η`: trust region radius for the next iteration.
- `next_λ`: soft penalty weight for the next iteration.
"""
function update_rule!(spbm::Subproblem)::Tuple{
        SubproblemSolution,
        RealTypes,
        RealTypes}

    # Extract values and relevant data
    pars = spbm.def.pars
    traj = spbm.def.traj
    t = spbm.def.common.t_grid
    sol = spbm.sol
    ref = spbm.ref
    N = pars.N
    ρ = sol.ρ
    η = spbm.η
    λ = spbm.λ
    κ = spbm.κ
    ρ0 = pars.ρ_0
    ρ1 = pars.ρ_1
    λ_init = pars.λ_init
    γ_fail = pars.γ_fail
    β_sh = pars.β_sh
    β_gr = pars.β_gr
    η_lb = pars.η_lb
    η_ub = pars.η_ub

    # Tolerances for checking trust region and soft-penalized constraint
    # satisfaction
    tr_buffer = 1e-3
    c_buffer = 1e-3

    # Compute trust region constraint satisfaction
    tr = trust_region_cost(sol.xd, sol.p, spbm, :nonconvex)
    trust_viol = any(tr.>tr_buffer)

    # Compute state and nonlinear path constraint satisfaction
    if !trust_viol
        feasible = true
        try
            # Check with respect to the convex state constraints
            if !isnothing(traj.X)
                for k = 1:N
                    cone_indicators = traj.X(t[k], k, sol.xd[:, k], sol.p)
                    for q in cone_indicators
                        if any(q.>c_buffer)
                            error("Convex state constraint violated")
                        end
                    end
                end
            end

            # Check with respect to the nonconvex path constraints
            if !isnothing(traj.s)
                for k = 1:N
                    s = traj.s(t[k], k, sol.xd[:, k], sol.p)
                    if any(s.>c_buffer)
                        error("Nonconvex path constraint violated")
                    end
                end
            end
        catch
            feasible = false
        end
    end

    # Apply update logic
    if trust_viol
        # Trust region constraint violated
        next_η = η
        next_ref = ref
        next_λ = γ_fail*λ
        sol.tr_update = ""
        sol.λ_update = "G"
        sol.reject = true
    else
        if ρ<ρ1
            if ρ<ρ0
                # Excellent cost and dynamics agreement
                next_η = min(η_ub, β_gr*η)
                next_ref = sol
                sol.tr_update = "G"
                sol.reject = false
            elseif ρ0<=ρ
                # Good cost and dynamics agreement
                next_η = η
                next_ref = sol
                sol.tr_update = ""
                sol.reject = false
            end
            # Update soft penalty weight
            if feasible
                next_λ = λ_init
                sol.λ_update = (next_λ<λ) ? "S" : ""
            else
                next_λ = γ_fail*λ
                sol.λ_update = "G"
            end
        else
            # Poor cost and dynamics agreement
            next_η = max(η_lb, η/β_sh)
            next_ref = ref
            next_λ = λ
            sol.tr_update = "S"
            sol.reject = true
            sol.λ_update = ""
        end
    end

    # Ensure that the trust region sequence goes to zero
    if κ < 1
        next_η *= κ
        # Use a * to indicate that shrinking is going on
        if length(sol.tr_update)==0
            sol.tr_update = " *"
        else
            sol.tr_update = sol.tr_update * "*"
        end
    end

    return next_ref, next_η, next_λ
end

"""
    cost_improvement_percent(J_new, J_old)

Compute the relative cost improvement (as a string to be put into a table).

# Arguments
- `J_new`: next cost.
- `J_old`: old cost.

# Returns
- `ΔJ`: the relative cost improvement.
"""
function cost_improvement_percent(
        J_new::RealTypes,
        J_old::RealTypes
)::String
    if isnan(J_old)
        ΔJ = ""
    else
        ΔJ = (J_old-J_new)/abs(J_old)*100
        _ΔJ = @sprintf("%.2f", ΔJ)
        if length(_ΔJ)>8
            fmt = string("%.", (ΔJ>0) ? 2 : 1, "e")
            ΔJ = @eval @sprintf($fmt, $ΔJ)
        else
            ΔJ = _ΔJ
        end
    end

    return ΔJ
end

"""
    print_info(spbm[, err])

Print command line info message.

# Arguments
- `spbm`: the subproblem that was solved.
- `err`: (optional) a GuSTO-specific error message.
"""
function print_info(
        spbm::Subproblem,
        err::Union{Nothing, SCPError}=nothing
)::Nothing

    # Convenience variables
    sol = spbm.sol
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
        ρ = !isnan(sol.ρ) ? @sprintf("%.2f", sol.ρ) : ""
        ρ = (length(ρ)>8) ? @sprintf("%.1e", sol.ρ) : ρ
        ΔJ = (!isnan(ref.J_aug) && sol.reject) ? "" :
            cost_improvement_percent(sol.J_aug, ref.J_aug)

        # Associate values with columns
        assoc = Dict(:iter => spbm.iter,
                     :status => status,
                     :cost => sol.J_aug,
                     :dx => max_dxh,
                     :du => max_duh,
                     :dp => max_dph,
                     :dynfeas => sol.feas ? "T" : "F",
                     :tr => spbm.η,
                     :soft => spbm.λ,
                     :ρ => ρ,
                     :cost_improv => ΔJ,
                     :dtr => sol.tr_update,
                     :dλ => sol.λ_update,
                     :rej => sol.reject ? "x" : "")

        print(assoc, table)
    end

    overhead!(spbm)

    return nothing
end
