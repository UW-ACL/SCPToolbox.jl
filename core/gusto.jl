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

using Printf

include("../utils/types.jl")
include("problem.jl")
include("scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Structure holding the GuSTO algorithm parameters."""
struct GuSTOParameters <: SCPParameters
    N::T_Int          # Number of temporal grid nodes
    Nsub::T_Int       # Number of subinterval integration time nodes
    iter_max::T_Int   # Maximum number of iterations
    ω::T_Real         # Dynamics virtual control weight
    λ_init::T_Real    # Initial soft penalty weight
    λ_max::T_Real     # Maximum soft penalty weight
    ρ_0::T_Real       # Trust region update threshold (lower, good solution)
    ρ_1::T_Real       # Trust region update threshold (upper, bad solution)
    β_sh::T_Real      # Trust region shrinkage factor
    β_gr::T_Real      # Trust region growth factor
    γ_fail::T_Real    # Soft penalty weight growth factor
    η_init::T_Real    # Initial trust region radius
    η_lb::T_Real      # Minimum trust region radius
    η_ub::T_Real      # Maximum trust region radius
    μ::T_Real         # Exponential shrink rate for trust region
    iter_μ::T_Real    # Iteration at which to apply trust region shrink
    ε_abs::T_Real     # Absolute convergence tolerance
    ε_rel::T_Real     # Relative convergence tolerance
    feas_tol::T_Real  # Dynamic feasibility tolerance
    pen::T_Symbol     # Penalty type (:quad, :softplus)
    hom::T_Real       # Homotopy parameter to use when pen==:softplus
    q_tr::T_Real      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::T_Real    # Stopping criterion norm
    solver::Module    # The numerical solver to use for the subproblems
    solver_opts::Dict{T_String, Any} # Numerical solver options
end

""" GuSTO subproblem solution."""
mutable struct GuSTOSubproblemSolution <: SCPSubproblemSolution
    iter::T_Int          # GuSTO iteration number
    # >> Discrete-time rajectory <<
    xd::T_RealMatrix     # States
    ud::T_RealMatrix     # Inputs
    p::T_RealVector      # Parameter vector
    # >> Virtual control terms <<
    vd::T_RealMatrix     # Dynamics virtual control
    # >> Cost values <<
    J::T_Real            # The original cost
    J_st::T_Real         # The state constraint soft penalty
    J_tr::T_Real         # The trust region soft penalty
    J_vc::T_Real         # The virtual control soft penalty
    J_aug::T_Real        # Overall nonlinear cost
    L::T_Real            # J *linearized* about reference solution
    L_st::T_Real         # J_st *linearized* about reference solution
    L_aug::T_Real        # Overall convex cost
    # >> Trajectory properties <<
    status::T_ExitStatus # Numerical optimizer exit status
    feas::T_Bool         # Dynamic feasibility flag
    defect::T_RealMatrix # "Defect" linearization accuracy metric
    deviation::T_Real    # Deviation from reference trajectory
    unsafe::T_Bool       # Indicator that the solution is unsafe to use
    cost_error::T_Real   # Cost error committed
    dyn_error::T_Real    # Cumulative dynamics error committed
    ρ::T_Real            # Convexification performance metric
    tr_update::T_String  # Growth direction indicator for trust region
    λ_update::T_String   # Growth direction indicator for soft penalty weight
    reject::T_Bool       # Indicator whether GuSTO rejected this solution
    dyn::T_DLTV          # The dynamics
end
const T_GuSTOSubSol = GuSTOSubproblemSolution # Alias

""" Subproblem definition in JuMP format for the convex numerical optimizer."""
mutable struct GuSTOSubproblem <: SCPSubproblem
    iter::T_Int                  # GuSTO iteration number
    mdl::Model                   # The optimization problem handle
    algo::T_String               # SCP and convex algorithms used
    # >> Algorithm parameters <<
    def::SCPProblem              # The GuSTO algorithm definition
    λ::T_Real                    # Soft penalty weight
    η::T_Real                    # Trust region radius
    κ::T_Real                    # Trust region shrinking multiplier
    # >> Reference and solution trajectories <<
    sol::Union{T_GuSTOSubSol, Missing} # Solution trajectory
    ref::Union{T_GuSTOSubSol, Missing} # Reference trajectory
    # >> Cost function <<
    L::T_Objective               # The original cost
    L_st::T_Objective            # The state constraint soft penalty
    L_tr::T_Objective            # The trust region soft penalty
    L_vc::T_Objective            # The virtual control soft penalty
    L_aug::T_Objective           # Overall cost
    # >> Scaled variables <<
    xh::T_OptiVarMatrix          # Discrete-time states
    uh::T_OptiVarMatrix          # Discrete-time inputs
    ph::T_OptiVarVector          # Parameter
    # >> Physical variables <<
    x::T_OptiVarMatrix           # Discrete-time states
    u::T_OptiVarMatrix           # Discrete-time inputs
    p::T_OptiVarVector           # Parameters
    # >> Virtual control (never scaled) <<
    vd::T_OptiVarMatrix          # Dynamics virtual control
    # >> Statistics <<
    nvar::T_Int                    # Total number of decision variables
    ncons::Dict{T_Symbol, Any}     # Number of constraints
    timing::Dict{T_Symbol, T_Real} # Runtime profiling
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    GuSTOProblem(pars, traj)

Construct the GuSTO problem definition.

# Arguments
- `pars`: GuSTO algorithm parameters.
- `traj`: the underlying trajectory optimization problem.

# Returns
- `pbm`: the problem structure ready for being solved by GuSTO.
"""
function GuSTOProblem(pars::GuSTOParameters,
                      traj::TrajectoryProblem)::SCPProblem

    table = T_Table([
        # Iteration count
        (:iter, "k", "%d", 2),
        # Solver status
        (:status, "status", "%s", 8),
        # Maximum dynamics virtual control element
        (:maxvd, "vd", "%.0e", 5),
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
        (:rej, "rej", "%s", 5)])

    pbm = SCPProblem(pars, traj, table)

    return pbm
end

"""
    GuSTOSubproblem(pbm[, iter, λ, η, ref])

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
function GuSTOSubproblem(pbm::SCPProblem,
                         iter::T_Int=0,
                         λ::T_Real=1e4,
                         η::T_Real=1.0,
                         ref::Union{T_GuSTOSubSol,
                                    Missing}=missing)::GuSTOSubproblem

    # Statistics
    timing = Dict(:formulate => time_ns(), :total => time_ns())
    nvar = 0
    ncons = Dict()

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
    mdl = Model()
    set_optimizer(mdl, solver.Optimizer)
    for (key,val) in solver_opts
        set_optimizer_attribute(mdl, key, val)
    end
    cvx_algo = string(pars.solver)
    algo = @sprintf("GuSTO (backend: %s)", cvx_algo)

    sol = missing # No solution associated yet with the subproblem

    # Cost function
    L = missing
    L_st = missing
    L_tr = missing
    L_vc = missing
    L_aug = missing

    # Decision variables (scaled)
    xh = @variable(mdl, [1:nx, 1:N], base_name="xh")
    uh = @variable(mdl, [1:nu, 1:N], base_name="uh")
    ph = @variable(mdl, [1:np], base_name="ph")

    # Physical decision variables
    x = scale.Sx*xh.+scale.cx
    u = scale.Su*uh.+scale.cu
    p = scale.Sp*ph.+scale.cp
    vd = @variable(mdl, [1:size(_E, 2), 1:N-1], base_name="vd")

    # Trust region shrink factor
    κ = (iter < pars.iter_μ) ? 1.0 : pars.μ^(1+iter-pars.iter_μ)

    spbm = GuSTOSubproblem(iter, mdl, algo, pbm, λ, η, κ, sol, ref, L, L_st,
                           L_tr, L_vc, L_aug, xh, uh, ph, x, u, p, vd, nvar,
                           ncons, timing)

    return spbm
end

"""
    GuSTOSubproblemSolution(x, u, p, iter, pbm)

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
function GuSTOSubproblemSolution(
    x::T_RealMatrix,
    u::T_RealMatrix,
    p::T_RealVector,
    iter::T_Int,
    pbm::SCPProblem)::T_GuSTOSubSol

    # Parameters
    N = pbm.pars.N
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    nv = size(pbm.common.E, 2)

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
    dyn = T_DLTV(nx, nu, np, nv, N)

    vd = T_RealMatrix(undef, 0, N)

    J = NaN
    J_st = NaN
    J_tr = NaN
    J_vc = NaN
    J_aug = NaN
    L = NaN
    L_st = NaN
    L_aug = NaN

    subsol = GuSTOSubproblemSolution(iter, x, u, p, vd, J, J_st, J_tr, J_vc,
                                     J_aug, L, L_st, L_aug, status, feas,
                                     defect, deviation, unsafe, cost_error,
                                     dyn_error, ρ, tr_update, λ_update, reject,
                                     dyn)

    # Compute the DLTV dynamics around this solution
    _scp__discretize!(subsol, pbm)

    return subsol
end

"""
    GuSTOSubproblemSolution(spbms)

Construct subproblem solution from a subproblem object. Expects that the
subproblem argument is a solved subproblem (i.e. one to which numerical
optimization has been applied).

# Arguments
- `spbm`: the subproblem structure.

# Returns
- `sol`: subproblem solution.
"""
function GuSTOSubproblemSolution(spbm::GuSTOSubproblem)::T_GuSTOSubSol
    # Extract the discrete-time trajectory
    x = value.(spbm.x)
    u = value.(spbm.u)
    p = value.(spbm.p)

    # Form the partly uninitialized subproblem
    sol = GuSTOSubproblemSolution(x, u, p, spbm.iter, spbm.def)

    # Save the virtual control values and penalty terms
    sol.vd = value.(spbm.vd)

    # Save the optimal cost values
    sol.J = _gusto__original_cost(x, u, p, spbm, :nonconvex)
    sol.J_st = _gusto__state_penalty_cost(x, p, spbm, :nonconvex)
    sol.J_tr = value.(spbm.L_tr)
    sol.J_vc = value.(spbm.L_vc)
    sol.J_aug = sol.J+sol.J_st+sol.J_tr+sol.J_vc
    sol.L = value.(spbm.L)
    sol.L_st = value.(spbm.L_st)
    sol.L_aug = value.(spbm.L_aug)

    return sol
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    gusto_solve(pbm)

Apply the GuSTO algorithm to solve the trajectory generation problem.

# Arguments
- `pbm`: the trajectory problem to be solved.

# Returns
- `sol`: the GuSTO solution structure.
- `history`: GuSTO iteration data history.
"""
function gusto_solve(pbm::SCPProblem)::Tuple{Union{SCPSolution,
                                                   T_GuSTOSubSol,
                                                   Nothing},
                                             SCPHistory}
    # ..:: Initialize ::..

    λ = pbm.pars.λ_init
    η = pbm.pars.η_init
    ref = _gusto__generate_initial_guess(pbm)

    history = SCPHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        spbm = GuSTOSubproblem(pbm, k, λ, η, ref)

        _gusto__add_cost!(spbm)
        _scp__add_dynamics!(spbm; relaxed=false)
        _scp__add_convex_input_constraints!(spbm)
        _scp__add_bcs!(spbm; relaxed=false)

        _scp__save!(history, spbm)

        try
            # >> Solve the subproblem <<
            _scp__solve_subproblem!(spbm)

            # "Emergency exit" the GuSTO loop if something bad happened
            # (e.g. numerical problems)
            if _scp__unsafe_solution(spbm)
                _gusto__print_info(spbm)
                break
            end

            # >> Check stopping criterion <<
            stop = _gusto__check_stopping_criterion!(spbm)
            if stop
                _gusto__print_info(spbm)
                break
            end

            # >> Update trust region <<
            ref, η, λ = _gusto__update_trust_region!(spbm)
        catch e
            isa(e, SCPError) || rethrow(e)
            _gusto__print_info(spbm, e)
            break
        end

        # >> Print iteration info <<
        _gusto__print_info(spbm)
    end

    reset(pbm.common.table)

    # ..:: Save solution ::..
    sol = SCPSolution(history)

    return sol, history
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    _gusto__generate_initial_guess(pbm)

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to a GuSTOSubproblemSolution structure.

# Arguments
- `pbm`: the GuSTO problem structure.

# Returns
- `guess`: the initial guess.
"""
function _gusto__generate_initial_guess(
    pbm::SCPProblem)::T_GuSTOSubSol

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    _scp__correct_convex!(x, u, p, pbm, :GuSTOSubproblem)
    guess = GuSTOSubproblemSolution(x, u, p, 0, pbm)

    return guess
end

"""
    _gusto__add_cost!(spbm)

Define the subproblem cost function.

# Arguments
- `spbm`: the subproblem definition.
"""
function _gusto__add_cost!(spbm::GuSTOSubproblem)::Nothing

    # Variables and parameters
    x = spbm.x
    u = spbm.u
    p = spbm.p
    vd = spbm.vd

    # Compute the cost components
    spbm.L = _gusto__original_cost(x, u, p, spbm)
    spbm.L_st = _gusto__state_penalty_cost(x, p, spbm)
    spbm.L_tr = _gusto__trust_region_cost(x, p, spbm)
    spbm.L_vc = _gusto__virtual_control_cost(vd, spbm)

    # Overall cost
    spbm.L_aug = spbm.L+spbm.L_st+spbm.L_tr+spbm.L_vc

    # Associate cost function with the model
    set_objective_function(spbm.mdl, spbm.L_aug)
    set_objective_sense(spbm.mdl, MOI.MIN_SENSE)

    return nothing
end

"""
    _gusto__original_cost(x, u, p, spbm[, mode])

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
function _gusto__original_cost(x::T_OptiVarMatrix,
                               u::T_OptiVarMatrix,
                               p::T_OptiVarVector,
                               spbm::GuSTOSubproblem,
                               mode::T_Symbol=:convex)::T_Objective
    # Parameters
    pars = spbm.def.pars
    traj = spbm.def.traj
    ref = spbm.ref
    N = pars.N
    t = spbm.def.common.t_grid
    if mode!=:convex
        xb = ref.xd
        ub = ref.ud
        pb = ref.p
    end

    # Terminal cost
    xf = @last(x)
    cost_term = isnothing(traj.φ) ? 0.0 : traj.φ(xf, p)

    # Integrated running cost
    cost_run_integrand = Vector{T_Objective}(undef, N)
    no_running_cost = (isnothing(traj.S) &&
                       isnothing(traj.ℓ) &&
                       isnothing(traj.g))
    for k = 1:N
        if no_running_cost
            @k(cost_run_integrand) = 0.0
        elseif mode==:convex
            Γk = 0.0
            nz = !isnothing # "nonzero" alias
            if nz(traj.S)
                if traj.S_cvx
                    Γk += @k(u)'*traj.S(p)*@k(u)
                else
                    uSu = @k(ub)'*traj.S(pb)*@k(ub)
                    ∇u_uSu = 2*traj.S(pb)*@k(ub)
                    ∇p_S = traj.dSdp(pb)
                    ∇p_uSu = [@k(ub)'*∇p_S[i]*@k(ub) for i=1:traj.np]
                    du = @k(u)-@k(ub)
                    dp = p-pb
                    uSu1 = uSu+∇u_uSu'*du+∇p_uSu.*dp
                    Γk += uSu1
                end
            end
            if nz(traj.ℓ)
                if traj.ℓ_cvx
                    Γk += @k(u)'*traj.ℓ(@k(x), p)
                else
                    uℓ = @k(ub)'*traj.ℓ(@k(xb), pb)
                    ∇u_uℓ = traj.ℓ(@k(xb), pb)
                    ∇x_uℓ = nz(traj.dℓdx) ? traj.dℓdx(@k(xb), pb)'*@k(ub) :
                        zeros(traj.nx)
                    ∇p_uℓ = nz(traj.dℓdp) ? traj.dℓdp(@k(xb), pb)'*@k(ub) :
                        zeros(traj.np)
                    du = @k(u)-@k(ub)
                    dx = @k(x)-@k(xb)
                    dp = p-pb
                    uℓ1 = uℓ+∇u_uℓ'*du+∇x_uℓ'*dx+∇p_uℓ'*dp
                    Γk += uℓ1
                end
            end
            if nz(traj.g)
                if traj.g_cvx
                    Γk += traj.g(@k(x), p)
                else
                    g = traj.g(@k(xb), pb)
                    ∇x_g = nz(traj.dgdx) ? traj.dgdx(@k(xb), pb) :
                        zeros(traj.nx)
                    ∇p_g = nz(traj.dgdp) ? traj.dgdp(@k(xb), pb) :
                        zeros(traj.np)
                    dx = @k(x)-@k(xb)
                    dp = p-pb
                    g1 = g+∇x_g'*dx+∇p_g'*dp
                    Γk += g1
                end
            end
            @k(cost_run_integrand) = Γk
        else
            Γk = 0.0
            Γk += !isnothing(traj.S) ? @k(u)'*traj.S(p)*@k(u) : 0.0
            Γk += !isnothing(traj.ℓ) ? @k(u)'*traj.ℓ(@k(x), p) : 0.0
            Γk += !isnothing(traj.g) ? traj.g(@k(x), p) : 0.0
            @k(cost_run_integrand) = Γk
        end
    end
    cost_run = trapz(cost_run_integrand, t)

    # Overall original cost
    cost = cost_term+cost_run

    return cost
end

"""
    _gusto__state_penalty_cost(x, p, spbm[, mode])

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
function _gusto__state_penalty_cost(x::T_OptiVarMatrix,
                                    p::T_OptiVarVector,
                                    spbm::GuSTOSubproblem,
                                    mode::T_Symbol=:convex)::T_Objective

    # Parameters
    pbm = spbm.def
    pars = pbm.pars
    traj = pbm.traj
    N = pars.N
    t = pbm.common.t_grid
    nx = traj.nx
    np = traj.np
    if mode==:convex
        ref = spbm.ref
        xb = ref.xd
        pb = ref.p
        dx = x-xb
        dp = p-pb
    end

    cost_st = 0.0

    # ..:: Convex state constraints ::..
    if !isnothing(traj.X)
        cost_soft_X = Vector{T_Objective}(undef, N)
        for k = 1:N
            in_X = traj.X(@k(t), k, @k(x), p)
            cost_soft_X[k] = 0.0
            for cone in in_X
                ρ = get_conic_constraint_indicator!(spbm.mdl, cone)
                for ρi in ρ
                    cost_soft_X[k] += _gusto__soft_penalty(spbm, ρi)
                end
            end
        end
        cost_st += trapz(cost_soft_X, t)
    end

    # ..:: Nonconvex path constraints ::..
    if !isnothing(traj.s)
        cost_soft_s = Vector{T_Objective}(undef, N)
        for k = 1:N
            cost_soft_s[k] = 0.0
            if mode==:convex
                tkxp = (@k(t), k, @k(xb), pb)
                s = traj.s(tkxp...)
                ns = length(s)
                dsdx = !isnothing(traj.C) ? traj.C(tkxp...) : zeros(ns, nx)
                dsdp = !isnothing(traj.G) ? traj.G(tkxp...) : zeros(ns, np)
                for i = 1:ns
                    cost_soft_s[k] += _gusto__soft_penalty(
                        spbm, s[i], dsdx[i, :], dsdp[i, :], @k(dx), dp)
                end
            else
                s = traj.s(@k(t), k, @k(x), p)
                ns = length(s)
                for i = 1:ns
                    cost_soft_s[k] += _gusto__soft_penalty(spbm, s[i])
                end
            end
        end
        cost_st += trapz(cost_soft_s, t)
    end

    return cost_st
end

"""
    _gusto__soft_penalty(spbm, f[, dfdx, dfdp, dx, dp])

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
- `h`: penalization function value.
"""
function _gusto__soft_penalty(
    spbm::GuSTOSubproblem,
    f::T_OptiVar,
    dfdx::Union{T_OptiVar, Nothing}=nothing,
    dfdp::Union{T_OptiVar, Nothing}=nothing,
    dx::Union{T_OptiVar, Nothing}=nothing,
    dp::Union{T_OptiVar, Nothing}=nothing)::T_Objective

    # Parameters
    pars = spbm.def.pars
    traj = spbm.def.traj
    penalty = pars.pen
    hom = pars.hom
    λ = spbm.λ
    linearized = !isnothing(dfdx) || !isnothing(dfdp)
    mode = (typeof(f)!=T_Real ||
            (linearized && (typeof(dx)!=T_RealVector ||
                            typeof(dp)!=T_RealVector))) ? :jump : :numerical

    # Get linearized version of the quantity being penalized, if applicable
    if linearized
        dfdx = !isnothing(dfdx) ? dfdx : zeros(traj.nx)
        dfdp = !isnothing(dfdp) ? dfdp : zeros(traj.np)
        dx = !isnothing(dx) ? dx : zeros(traj.nx)
        dp = !isnothing(dp) ? dp : zeros(traj.np)
        f = f+dfdx'*dx+dfdp'*dp
    end

    # Compute the function value
    # The possibilities are:
    #   (:quad)      h(f(x, p)) = λ*(max(0, f(x, p)))^2
    #   (:softplus)  h(f(x, p)) = λ*log(1+exp(hom*f(x, p)))/hom
    acc! = add_conic_constraint!
    Cone = T_ConvexConeConstraint
    if penalty==:quad
        # ..:: Quadratic penalty ::..
        if mode==:numerical
            h = (max(0.0, f))^2
        else
            u = @variable(spbm.mdl, base_name="u")
            v = @variable(spbm.mdl, base_name="v")
            acc!(spbm.mdl, Cone(-u, :nonpos))
            acc!(spbm.mdl, Cone(f+u-v, :nonpos))
            h = v^2
        end
    else
        # ..:: Log-sum-exp penalty ::..
        if mode==:numerical
            F = [0, f]
            h = logsumexp(F; t=hom)
        else
            u = @variable(spbm.mdl, base_name="u")
            v = @variable(spbm.mdl, base_name="v")
            w = @variable(spbm.mdl, base_name="w")
            acc!(spbm.mdl, Cone(vcat(-w, 1, u), :exp))
            acc!(spbm.mdl, Cone(vcat(hom*f-w, 1, v), :exp))
            acc!(spbm.mdl, Cone(u+v-1, :nonpos))
            h = w/hom
        end
    end
    h *= λ

    return h
end

"""
    _gusto__trust_region_cost(x, p, spbm[, mode; raw])

Compute the trust region constraint soft penalty. This function has two
"modes": the (default) convex mode computes the convex version of the cost
(where all non-convexity has been convexified), while the nonconvex mode
computes the fully nonlinear cost.

# Arguments
- `x`: the discrete-time state trajectory.
- `p`: the parameter vector.
- `spbm`: the subproblem structure.
- `mode`: (optional) either :convex (default) or :nonconvex.

# Keywords
- `raw`: (optional) the value false (default) means to integrate and return the
  integrated penalty. Otherwise, if true, then return the actual trust region
  left-hand sides (which should be <=0 if the trust region constraints are
  satisfied).

# Returns
- `cost_tr`: if raw is false, the trust region soft penalty cost; or
- `tr`: if raw is true, the trust regions left-hand sides (which should all be
  <=0 if the trust region constraints are satisfied).
"""
function _gusto__trust_region_cost(x::T_OptiVarMatrix,
                                   p::T_OptiVarVector,
                                   spbm::GuSTOSubproblem,
                                   mode::T_Symbol=:convex;
                                   raw::T_Bool=false)::Union{T_Objective,
                                                             T_RealVector}

    # Parameters
    pars = spbm.def.pars
    scale = spbm.def.common.scale
    q = pars.q_tr
    N = pars.N
    η = spbm.η
    sqrt_η = sqrt(η)
    t = spbm.def.common.t_grid
    xh = scale.iSx*(x.-scale.cx)
    ph = scale.iSp*(p-scale.cp)
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)
    dx = xh-xh_ref
    dp = ph-ph_ref
    tr = (mode==:convex) ? @variable(spbm.mdl, [1:N], base_name="tr") :
        T_RealVector(undef, N)

    # Integrated running cost
    if !raw
        cost_tr_integrand = Vector{T_Objective}(undef, N)
    end
    q2cone = Dict(1 => :l1, 2 => :soc, 4 => :soc, Inf => :linf)
    cone = q2cone[q]
    if mode==:convex
        C = T_ConvexConeConstraint
        acc! = add_conic_constraint!
        dx_lq = @variable(spbm.mdl, [1:N], base_name="dx_lq")
        dp_lq = @variable(spbm.mdl, base_name="dp_lq")
        acc!(spbm.mdl, C(vcat(dp_lq, dp), cone))
    else
        dp_lq = norm(dp, q)
    end
    for k = 1:N
        if mode==:convex
            acc!(spbm.mdl, C(vcat(@k(dx_lq), @k(dx)), cone))
            if q==4
                w = @variable(spbm.mdl, base_name="w")
                acc!(spbm.mdl, C(vcat(w, @k(dx_lq), dp_lq), :soc))
                acc!(spbm.mdl, C(vcat(w, η+@k(tr), 1), :geom))
            else
                acc!(spbm.mdl, C(@k(dx_lq)+dp_lq-(η+@k(tr)), :nonpos))
            end
        else
            dx_lq = norm(@k(dx), q)
            w = (q==4) ? 2 : 1
            @k(tr) = dx_lq^w+dp_lq^w-η
        end
        if !raw
            @k(cost_tr_integrand) = _gusto__soft_penalty(spbm, @k(tr))
        end
    end
    if !raw
        cost_tr = trapz(cost_tr_integrand, t)
    end

    return (raw) ? tr : cost_tr
end

"""
    _gusto__virtual_control_cost(vd, spbm)

Compute the virtual control penalty.

# Arguments
- `vd`: the discrete-time dynamics virtual control trajectory.
- `spbm`: the subproblem structure.

# Returns
- `cost_vc`: the virtual control penalty cost.
"""
function _gusto__virtual_control_cost(vd::T_OptiVarMatrix,
                                      spbm::GuSTOSubproblem)::T_Objective

    # Parameters
    pars = spbm.def.pars
    ω = pars.ω
    N = pars.N
    t = spbm.def.common.t_grid
    E = spbm.ref.dyn.E
    vc_l1 = @variable(spbm.mdl, [1:N-1], base_name="vc_l1")

    # Integrated running cost
    cost_vc_integrand = Vector{T_Objective}(undef, N)
    cost_vc_integrand[end] = 0.0
    for k = 1:N-1
        vck_l1 = @k(vc_l1)
        C = T_ConvexConeConstraint(vcat(vck_l1, @k(E)*@k(vd)), :l1)
        add_conic_constraint!(spbm.mdl, C)
        @k(cost_vc_integrand) = ω*vck_l1
    end
    cost_vc = trapz(cost_vc_integrand, t)

    return cost_vc
end

"""
    _gusto__check_stopping_criterion!(spbm)

Check if stopping criterion is triggered.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `stop`: true if stopping criterion holds.
"""
function _gusto__check_stopping_criterion!(spbm::GuSTOSubproblem)::T_Bool

    # Extract values
    pbm = spbm.def
    ref = spbm.ref
    sol = spbm.sol
    ε_abs = pbm.pars.ε_abs
    ε_rel = pbm.pars.ε_rel
    λ = spbm.λ
    λ_max = pbm.pars.λ_max

    # Compute solution deviation from reference
    sol.deviation = _scp__solution_deviation(spbm)

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
    _gusto__update_trust_region!(spbm)

Compute the new reference, trust region, and soft penalty.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `next_ref`: reference trajectory for the next iteration.
- `next_η`: trust region radius for the next iteration.
- `next_λ`: soft penalty weight for the next iteration.
"""
function _gusto__update_trust_region!(
    spbm::GuSTOSubproblem)::Tuple{GuSTOSubproblemSolution,
                                  T_Real,
                                  T_Real}

    # Parameters
    pbm = spbm.def
    traj = pbm.traj
    N = pbm.pars.N
    Nsub = pbm.pars.Nsub
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
    Δf = T_RealVector(undef, N)
    dxdt = T_RealVector(undef, N)
    for k = 1:N
        f = traj.f(@k(t), k, @k(xb), @k(ub), pb)
        A = traj.A(@k(t), k, @k(xb), @k(ub), pb)
        B = traj.B(@k(t), k, @k(xb), @k(ub), pb)
        F = traj.F(@k(t), k, @k(xb), @k(ub), pb)
        r = f-A*@k(xb)-B*@k(ub)-F*pb
        f_lin = A*@k(x)+B*@k(u)+F*p+r
        f_nl = traj.f(@k(t), k, @k(x), @k(u), p)
        @k(Δf) = norm(f_nl-f_lin)
        @k(dxdt) = norm(f_lin)
    end
    sol.dyn_error = trapz(Δf, t)
    dynamics_nrml = trapz(dxdt, t)

    # Convexification performance metric
    normalization_term = cost_nrml+dynamics_nrml
    sol.ρ = (sol.cost_error+sol.dyn_error)/normalization_term

    # Apply update rule
    next_ref, next_η, next_λ = _gusto__update_rule!(spbm)

    return next_ref, next_η, next_λ
end

"""
    _gusto__update_rule!(spbm)

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
function _gusto__update_rule!(
    spbm::GuSTOSubproblem)::Tuple{GuSTOSubproblemSolution,
                                  T_Real,
                                  T_Real}

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
    λ_max = pars.λ_max
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
    tr = _gusto__trust_region_cost(sol.xd, sol.p, spbm, :nonconvex; raw=true)
    trust_viol = any(tr.>tr_buffer)

    # Compute state and nonlinear path constraint satisfaction
    if !trust_viol
        feasible = true
        try
            # Check with respect to the convex state constraints
            if !isnothing(traj.X)
                for k = 1:N
                    in_X = traj.X(@k(t), k, @k(sol.xd), sol.p)
                    for cone in in_X
                        ind = get_conic_constraint_indicator!(spbm.mdl, cone)
                        if any(ind.>c_buffer)
                            error("Convex state constraint violated")
                        end
                    end
                end
            end

            # Check with respect to the nonconvex path constraints
            if !isnothing(traj.s)
                for k = 1:N
                    s = traj.s(@k(t), k, @k(sol.xd), sol.p)
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
    _gusto__print_info(spbm[, err])

Print command line info message.

# Arguments
- `spbm`: the subproblem that was solved.
- `err`: (optional) a GuSTO-specific error message.
"""
function _gusto__print_info(spbm::GuSTOSubproblem,
                            err::Union{Nothing, SCPError}=nothing)::Nothing

    # Convenience variables
    sol = spbm.sol
    ref = spbm.ref
    table = spbm.def.common.table

    if !isnothing(err)
        @printf "ERROR: %s, exiting\n" err.msg
    elseif _scp__unsafe_solution(sol)
        @printf "ERROR: unsafe solution (%s), exiting\n" sol.status
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
                     :maxvd => norm(sol.vd, Inf),
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

    _scp__overhead!(spbm)

    return nothing
end
