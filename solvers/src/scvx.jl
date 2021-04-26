#= SCvx algorithm data structures and methods.

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

include("scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Structure holding the SCvx algorithm parameters."""
struct SCvxParameters <: SCPParameters
    N::T_Int          # Number of temporal grid nodes
    Nsub::T_Int       # Number of subinterval integration time nodes
    iter_max::T_Int   # Maximum number of iterations
    λ::T_Real         # Virtual control weight
    ρ_0::T_Real       # Trust region update threshold (lower, bad solution)
    ρ_1::T_Real       # Trust region update threshold (middle, OK solution)
    ρ_2::T_Real       # Trust region update threshold (upper, good solution)
    β_sh::T_Real      # Trust region shrinkage factor
    β_gr::T_Real      # Trust region growth factor
    η_init::T_Real    # Initial trust region radius
    η_lb::T_Real      # Minimum trust region radius
    η_ub::T_Real      # Maximum trust region radius
    ε_abs::T_Real     # Absolute convergence tolerance
    ε_rel::T_Real     # Relative convergence tolerance
    feas_tol::T_Real  # Dynamic feasibility tolerance
    q_tr::T_Real      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::T_Real    # Stopping criterion norm
    solver::Module    # The numerical solver to use for the subproblems
    solver_opts::Dict{T_String, Any} # Numerical solver options
end

""" SCvx subproblem solution."""
mutable struct SCvxSubproblemSolution <: SCPSubproblemSolution
    iter::T_Int          # SCvx iteration number
    # >> Discrete-time rajectory <<
    xd::T_RealMatrix     # States
    ud::T_RealMatrix     # Inputs
    p::T_RealVector      # Parameter vector
    # >> Virtual control terms <<
    vd::T_RealMatrix     # Dynamics virtual control
    vs::T_RealMatrix     # Nonconvex constraints virtual control
    vic::T_RealVector    # Initial conditions virtual control
    vtc::T_RealVector    # Terminal conditions virtual control
    P::T_RealVector      # Virtual control penalty integrand terms
    Pf::T_RealVector     # Boundary condition virtual control penalty
    # >> Cost values <<
    L::T_Real            # The original cost
    L_pen::T_Real        # The virtual control penalty
    L_aug::T_Real        # Overall cost
    J_aug::T_Real        # Overall cost for nonlinear problem
    act_improv::T_Real   # Actual cost improvement
    pre_improv::T_Real   # Predicted cost improvement
    # >> Trajectory properties <<
    status::T_ExitStatus # Numerical optimizer exit status
    feas::T_Bool         # Dynamic feasibility flag
    defect::T_RealMatrix # "Defect" linearization accuracy metric
    deviation::T_Real    # Deviation from reference trajectory
    unsafe::T_Bool       # Indicator that the solution is unsafe to use
    ρ::T_Real            # Convexification performance metric
    tr_update::T_String  # Indicator of growth direction for trust region
    reject::T_Bool       # Indicator whether SCvx rejected this solution
    dyn::T_DLTV          # The dynamics
end

""" Subproblem definition in JuMP format for the convex numerical optimizer."""
mutable struct SCvxSubproblem <: SCPSubproblem
    iter::T_Int          # SCvx iteration number
    mdl::Model           # The optimization problem handle
    algo::T_String       # SCP and convex algorithms used
    # >> Algorithm parameters <<
    def::SCPProblem      # The SCvx problem definition
    η::T_Real            # Trust region radius
    # >> Reference and solution trajectories <<
    sol::Union{SCvxSubproblemSolution, Missing} # Solution trajectory
    ref::Union{SCvxSubproblemSolution, Missing} # Reference trajectory
    # >> Cost function <<
    L::T_Objective       # The original convex cost function
    L_pen::T_Objective   # The virtual control penalty
    L_aug::T_Objective   # Overall cost function
    # >> Scaled variables <<
    xh::T_OptiVarMatrix  # Discrete-time states
    uh::T_OptiVarMatrix  # Discrete-time inputs
    ph::T_OptiVarVector  # Parameter
    # >> Physical variables <<
    x::T_OptiVarMatrix   # Discrete-time states
    u::T_OptiVarMatrix   # Discrete-time inputs
    p::T_OptiVarVector   # Parameters
    # >> Virtual control (never scaled) <<
    vd::T_OptiVarMatrix  # Dynamics virtual control
    vs::T_OptiVarMatrix  # Nonconvex constraints virtual control
    vic::T_OptiVarVector # Initial conditions virtual control
    vtc::T_OptiVarVector # Terminal conditions virtual control
    # >> Other variables <<
    P::T_OptiVarVector   # Virtual control penalty
    Pf::T_OptiVarVector  # Boundary condition virtual control penalty
    # >> Statistics <<
    nvar::T_Int                    # Total number of decision variables
    ncons::Dict{T_Symbol, Any}     # Number of constraints
    timing::Dict{T_Symbol, T_Real} # Runtime profiling
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    SCvxProblem(pars, traj)

Construct the SCvx problem definition.

# Arguments
- `pars`: SCvx algorithm parameters.
- `traj`: the underlying trajectory optimization problem.

# Returns
- `pbm`: the problem structure ready for being solved by SCvx.
"""
function SCvxProblem(pars::SCvxParameters,
                     traj::TrajectoryProblem)::SCPProblem

    table = T_Table([
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
        # Trust region size
        (:tr, "η", "%.2f", 5),
        # Convexification performance metric
        (:ρ, "ρ", "%s", 9),
        # Predicted cost improvement (percent)
        (:pre_improv, "J-L %", "%.2f", 9),
        # Update direction for trust region radius (grow? shrink?)
        (:dtr, "Δη", "%s", 3),
        # Reject solution indicator
        (:rej, "rej", "%s", 5)])

    pbm = SCPProblem(pars, traj, table)

    return pbm
end

"""
    SCvxSubproblem(pbm[, iter, η, ref])

Constructor for an empty convex optimization subproblem. No cost or
constraints. Just the decision variables and empty associated parameters.

# Arguments
- `pbm`: the SCvx problem being solved.
- `iter`: (optional) SCvx iteration number.
- `η`: (optional) the trust region radius.
- `ref`: (optional) the reference trajectory.

# Returns
- `spbm`: the subproblem structure.
"""
function SCvxSubproblem(pbm::SCPProblem,
                        iter::T_Int=0,
                        η::T_Real=1.0,
                        ref::Union{SCvxSubproblemSolution,
                                   Missing}=missing)::SCvxSubproblem

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
    algo = @sprintf("SCvx (backend: %s)", cvx_algo)

    sol = missing # No solution associated yet with the subproblem

    # Cost
    L = missing
    L_pen = missing
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
    vs = T_RealMatrix(undef, 0, N)
    vic = T_RealVector(undef, 0)
    vtc = T_RealVector(undef, 0)

    # Other variables
    P = @variable(mdl, [1:N], base_name="P")
    Pf = @variable(mdl, [1:2], base_name="Pf")

    spbm = SCvxSubproblem(iter, mdl, algo, pbm, η, sol, ref, L, L_pen, L_aug,
                          xh, uh, ph, x, u, p, vd, vs, vic, vtc, P, Pf, nvar,
                          ncons, timing)

    return spbm
end

"""
    SCvxSubproblemSolution(x, u, p, iter, pbm)

Construct a subproblem solution from a discrete-time trajectory. This leaves
parameters of the solution other than the passed discrete-time trajectory
unset.

# Arguments
- `x`: discrete-time state trajectory.
- `u`: discrete-time input trajectory.
- `p`: parameter vector.
- `iter`: SCvx iteration number.
- `pbm`: the SCvx problem definition.

# Returns
- `subsol`: subproblem solution structure.
"""
function SCvxSubproblemSolution(
    x::T_RealMatrix,
    u::T_RealMatrix,
    p::T_RealVector,
    iter::T_Int,
    pbm::SCPProblem)::SCvxSubproblemSolution

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
    ρ = NaN
    tr_update = ""
    reject = false
    dyn = T_DLTV(nx, nu, np, nv, N)

    vd = T_RealMatrix(undef, 0, N)
    vs = T_RealMatrix(undef, 0, N)
    vic = T_RealVector(undef, 0)
    vtc = T_RealVector(undef, 0)
    P = zeros(N)
    Pf = zeros(2)

    L = NaN
    L_pen = NaN
    L_aug = NaN
    J_aug = NaN
    act_improv = NaN
    pre_improv = NaN

    subsol = SCvxSubproblemSolution(iter, x, u, p, vd, vs, vic, vtc, P, Pf, L,
                                    L_pen, L_aug, J_aug, act_improv,
                                    pre_improv, status, feas, defect,
                                    deviation, unsafe, ρ, tr_update, reject,
                                    dyn)

    # Compute the DLTV dynamics around this solution
    _scp__discretize!(subsol, pbm)

    # Nonlinear cost along this trajectory
    _scvx__solution_cost!(subsol, :nonlinear, pbm)

    return subsol
end

"""
    Signature

Construct subproblem solution from a subproblem object. Expects that the
subproblem argument is a solved subproblem (i.e. one to which numerical
optimization has been applied).

# Arguments
- `spbm`: the subproblem structure.

# Returns
- `sol`: subproblem solution.
"""
function SCvxSubproblemSolution(spbm::SCvxSubproblem)::SCvxSubproblemSolution
    # Extract the discrete-time trajectory
    x = value.(spbm.x)
    u = value.(spbm.u)
    p = value.(spbm.p)

    # Form the partly uninitialized subproblem
    sol = SCvxSubproblemSolution(x, u, p, spbm.iter, spbm.def)

    # Save the virtual control values and penalty terms
    sol.vd = value.(spbm.vd)
    sol.vs = value.(spbm.vs)
    sol.vic = value.(spbm.vic)
    sol.vtc = value.(spbm.vtc)
    sol.P = value.(spbm.P)
    sol.Pf = value.(spbm.Pf)

    # Save the optimal cost values
    sol.L = value(spbm.L)
    sol.L_pen = value(spbm.L_pen)
    sol.L_aug = value(spbm.L_aug)

    return sol
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    scvx_solve(pbm)

Apply the SCvx algorithm to solve the trajectory generation problem.

# Arguments
- `pbm`: the trajectory problem to be solved.

# Returns
- `sol`: the SCvx solution structure.
- `history`: SCvx iteration data history.
"""
function scvx_solve(pbm::SCPProblem)::Tuple{Union{SCPSolution, Nothing},
                                             SCPHistory}
    # ..:: Initialize ::..

    η = pbm.pars.η_init
    ref = _scvx__generate_initial_guess(pbm)

    history = SCPHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        spbm = SCvxSubproblem(pbm, k, η, ref)

        _scp__add_dynamics!(spbm)
        _scp__add_convex_state_constraints!(spbm)
        _scp__add_convex_input_constraints!(spbm)
        _scp__add_nonconvex_constraints!(spbm)
        _scp__add_bcs!(spbm)
        _scvx__add_trust_region!(spbm)
        _scvx__add_cost!(spbm)

        _scp__save!(history, spbm)

        try
            # >> Solve the subproblem <<
            _scp__solve_subproblem!(spbm)

            # "Emergency exit" the SCvx loop if something bad happened
            # (e.g. numerical problems)
            if _scp__unsafe_solution(spbm)
                _scvx__print_info(spbm)
                break
            end

            # >> Check stopping criterion <<
            stop = _scvx__check_stopping_criterion!(spbm)
            if stop
                _scvx__print_info(spbm)
                break
            end

            # >> Update trust region <<
            ref, η = _scvx__update_trust_region!(spbm)
        catch e
            isa(e, SCPError) || rethrow(e)
            _scvx__print_info(spbm, e)
            break
        end

        # >> Print iteration info <<
        _scvx__print_info(spbm)
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
    _scvx__generate_initial_guess(pbm)

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to an SCvxSubproblemSolution structure.

# Arguments
- `pbm`: the SCvx problem structure.

# Returns
- `guess`: the initial guess.
"""
function _scvx__generate_initial_guess(
    pbm::SCPProblem)::SCvxSubproblemSolution

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    _scp__correct_convex!(x, u, p, pbm, :SCvxSubproblem)
    guess = SCvxSubproblemSolution(x, u, p, 0, pbm)

    return guess
end

"""
    _scvx__add_trust_region!(spbm)

Add trust region constraint to the subproblem.

# Arguments
- `spbm`: the subproblem definition.
"""
function _scvx__add_trust_region!(spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    q = spbm.def.pars.q_tr
    scale = spbm.def.common.scale
    traj_pbm = spbm.def.traj
    nx = traj_pbm.nx
    nu = traj_pbm.nu
    np = traj_pbm.np
    η = spbm.η
    xh = spbm.xh
    uh = spbm.uh
    ph = spbm.ph
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    uh_ref = scale.iSu*(spbm.ref.ud.-scale.cu)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)

    # Measure the *scaled* state and input deviations
    dx = xh-xh_ref
    du = uh-uh_ref
    dp = ph-ph_ref

    # Trust region constraint
    q2cone = Dict(1 => :l1, 2 => :soc, 4 => :soc, Inf => :linf)
    cone = q2cone[q]
    C = T_ConvexConeConstraint
    acc! = add_conic_constraint!
    dx_lq = @variable(spbm.mdl, [1:N], base_name="dx_lq")
    du_lq = @variable(spbm.mdl, [1:N], base_name="du_lq")
    dp_lq = @variable(spbm.mdl, base_name="dp_lq")
    acc!(spbm.mdl, C(vcat(dp_lq, dp), cone))
    for k = 1:N
        acc!(spbm.mdl, C(vcat(@k(dx_lq), @k(dx)), cone))
        acc!(spbm.mdl, C(vcat(@k(du_lq), @k(du)), cone))
        if q==4
            w = @variable(spbm.mdl, base_name="w")
            acc!(spbm.mdl, C(vcat(w, @k(dx_lq), @k(du_lq), dp_lq), :soc))
            acc!(spbm.mdl, C(vcat(w, η, 1), :geom))
        else
            acc!(spbm.mdl, C(@k(dx_lq)+@k(du_lq)+dp_lq-η, :nonpos))
        end
    end

    return nothing
end

"""
    _scvx__add_cost!(spbm)

Define the subproblem cost function.

# Arguments
- `spbm`: the subproblem definition.
"""
function _scvx__add_cost!(spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Compute the cost components
    spbm.L = _scp__original_cost(x, u, p, spbm.def)
    _scvx__compute_linear_cost_penalty!(spbm)

    # Overall cost
    spbm.L_aug = spbm.L+spbm.L_pen

    # Associate cost function with the model
    set_objective_function(spbm.mdl, spbm.L_aug)
    set_objective_sense(spbm.mdl, MOI.MIN_SENSE)

    return nothing
end

"""
    _scvx__add_cost!(spbm)

Check if stopping criterion is triggered.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `stop`: true if stopping criterion holds.
"""
function _scvx__check_stopping_criterion!(spbm::SCvxSubproblem)::T_Bool

    # Extract values
    pbm = spbm.def
    ref = spbm.ref
    sol = spbm.sol
    ε_abs = pbm.pars.ε_abs
    ε_rel = pbm.pars.ε_rel

    # Compute solution deviation from reference
    sol.deviation = _scp__solution_deviation(spbm)

    # Check predicted cost improvement
    J_ref = _scvx__solution_cost!(ref, :nonlinear, pbm)
    L_sol = _scvx__solution_cost!(sol, :linear, pbm)
    sol.pre_improv = J_ref-L_sol
    pre_improv_rel = sol.pre_improv/abs(J_ref)

    # Compute stopping criterion
    stop = (spbm.iter>1 &&
            (sol.feas && (pre_improv_rel<=ε_rel || sol.deviation<=ε_abs)))

    return stop
end

"""
    _scvx__update_trust_region!(spbm)

Compute the new reference and trust region. Apply the trust region update rule
based on the most recent subproblem solution. This updates the trust region
radius, and selects either the current or the reference solution to act as the
next iteration's reference trajectory.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `next_ref`: reference trajectory for the next iteration.
- `next_η`: trust region radius for the next iteration.
"""
function _scvx__update_trust_region!(
    spbm::SCvxSubproblem)::Tuple{SCvxSubproblemSolution,
                                 T_Real}

    # Parameters
    pbm = spbm.def
    sol = spbm.sol
    ref = spbm.ref

    # Compute the actual cost improvement
    J_ref = _scvx__solution_cost!(ref, :nonlinear, pbm)
    J_sol = _scvx__solution_cost!(sol, :nonlinear, pbm)
    sol.act_improv = J_ref-J_sol

    # Convexification performance metric
    sol.ρ = sol.act_improv/sol.pre_improv

    # Apply update rule
    next_ref, next_η = _scvx__update_rule(spbm)

    return next_ref, next_η
end

"""
    _scvx__P(vd, vs)

Compute cost penalty at a particular instant.

This is the integrand of the overall cost penalty term for dynamics and
nonconvex constraint violation.

Note: **this function must match the penalty implemented in
_scvx__add_cost!()**.

# Arguments
- `vd`: inconsistency in the dynamics ("defect").
- `vs`: inconsistency in the nonconvex inequality constraints.

# Returns
- `P`: the penalty value.
"""
function _scvx__P(vd::T_RealVector, vs::T_RealVector)::T_Real
    P = norm(vd, 1)+norm(vs, 1)
    return P
end

"""
    _scvx__compute_linear_cost_penalty!(spbm)

Compute the subproblem cost virtual control penalty term.

# Arguments
- `spbm`: the subproblem definition.
"""
function _scvx__compute_linear_cost_penalty!(spbm::SCvxSubproblem)::Nothing
    # Variables and parameters
    N = spbm.def.pars.N
    λ = spbm.def.pars.λ
    t = spbm.def.common.t_grid
    E = spbm.ref.dyn.E
    P = spbm.P
    Pf = spbm.Pf
    vd = spbm.vd
    vs = spbm.vs
    vic = spbm.vic
    vtc = spbm.vtc

    # Compute virtual control penalty
    C = T_ConvexConeConstraint
    acc! = add_conic_constraint!
    for k = 1:N
        if k<N
            tmp = vcat(@k(P), @k(E)*@k(vd), @k(vs))
        else
            tmp = vcat(@k(P), @k(vs))
        end
        acc!(spbm.mdl, C(tmp, :l1))
    end
    acc!(spbm.mdl, C(vcat(@first(Pf), vic), :l1))
    acc!(spbm.mdl, C(vcat(@last(Pf), vtc), :l1))
    spbm.L_pen = trapz(λ*P, t)+sum(λ*Pf)

    return nothing
end

"""
    _scvx__actual_cost_penalty!(sol, pbm)

Compute the subproblem cost penalty based on actual constraint violation.

This computes the same cost penalty form as in the subproblem. However, instead
of using virtual control terms, it uses defects from nonlinear propagation of
the dynamics and the actual values of the nonconvex inequality constraints.

If the subproblem solution has already had this function called for it,
re-computation is skipped and the already computed value is returned.

# Arguments
- `sol`: the subproblem solution.
- `pbm`: the SCvx problem definition.
- `safe`: (optional) whether to check that the coded penalty function matches
  the optimization.

# Returns
- `pen`: the nonlinear cost penalty term.
"""
function _scvx__actual_cost_penalty!(
    sol::SCvxSubproblemSolution,
    pbm::SCPProblem)::T_Real

    # Values and parameters from the solution
    N = pbm.pars.N
    λ = pbm.pars.λ
    nx = pbm.traj.nx
    t = pbm.common.t_grid
    x = sol.xd
    u = sol.ud
    p = sol.p

    # Values from the solution
    δ = sol.defect
    gic = pbm.traj.gic(@first(sol.xd), sol.p)
    gtc = pbm.traj.gtc(@last(sol.xd), sol.p)

    # Integrate the nonlinear penalty term
    P = T_RealVector(undef, N)
    for k = 1:N
        δk = (k<N) ? @k(δ) : zeros(nx)
        sk = pbm.traj.s(@k(t), k, @k(x), @k(u), sol.p)
        sk = isempty(sk) ? [0.0] : sk
        @k(P) = λ*_scvx__P(δk, max.(sk, 0.0))
    end
    pen = trapz(P, t)+λ*(_scvx__P([0.0], gic)+_scvx__P([0.0], gtc))

    return pen
end

"""
    _scvx__solution_cost!(sol, kind, pbm)

Compute the linear or nonlinear overall associated with a solution.

# Arguments
- `sol`: the subproblem solution structure.
- `kind`: whether to compute the linear or nonlinear problem cost.
- `pbm`: the SCvx problem definition.

# Returns
- `cost`: the optimal cost associated with this solution.
"""
function _scvx__solution_cost!(
    sol::SCvxSubproblemSolution,
    kind::T_Symbol,
    pbm::SCPProblem)::T_Real

    if isnan(sol.L)
        sol.L = _scp__original_cost(sol.xd, sol.ud, sol.p, pbm)
    end

    if kind==:linear
        cost = sol.L
    else
        if isnan(sol.J_aug)
            J_orig = sol.L
            J_pen = _scvx__actual_cost_penalty!(sol, pbm)
            sol.J_aug = J_orig+J_pen
        end
        cost = sol.J_aug
    end

    return cost
end

"""
    _scvx__update_rule(spbm)

Apply the low-level SCvx trust region update rule. This computes the new trust
region value and reference trajectory, based on the obtained subproblem
solution.

# Arguments
- `spbm`: the subproblem definition.

# Returns
- `next_ref`: reference trajectory for the next iteration.
- `next_η`: trust region radius for the next iteration.
"""
function _scvx__update_rule(
    spbm::SCvxSubproblem)::Tuple{SCvxSubproblemSolution,
                                 T_Real}
    # Extract relevant data
    pars = spbm.def.pars
    sol = spbm.sol
    ref = spbm.ref
    ρ = sol.ρ
    ρ0 = pars.ρ_0
    ρ1 = pars.ρ_1
    ρ2 = pars.ρ_2
    β_sh = pars.β_sh
    β_gr = pars.β_gr
    η_lb = pars.η_lb
    η_ub = pars.η_ub
    η = spbm.η

    # Apply update logic
    # Prediction below means "prediction of cost improvement by the linearized
    # model"
    if ρ<ρ0
        # Very poor prediction
        next_η = max(η_lb, η/β_sh)
        next_ref = ref
        sol.tr_update = "S"
        sol.reject = true
    elseif ρ0<=ρ && ρ<ρ1
        # Mediocre prediction
        next_η = max(η_lb, η/β_sh)
        next_ref = sol
        sol.tr_update = "S"
        sol.reject = false
    elseif ρ1<=ρ && ρ<ρ2
        # Good prediction
        next_η = η
        next_ref = sol
        sol.tr_update = ""
        sol.reject = false
    else
        # Excellent prediction
        next_η = min(η_ub, β_gr*η)
        next_ref = sol
        sol.tr_update = "G"
        sol.reject = false
    end

    return next_ref, next_η
end

"""
    _scvx__mark_unsafe!(sol, err)

Mark a solution as unsafe to use.

# Arguments
- `sol`: subproblem solution.
- `err`: the SCvx error that occurred.
"""
function _scvx__mark_unsafe!(sol::SCvxSubproblemSolution,
                             err::SCPError)::Nothing
    sol.status = err.status
    sol.unsafe = true
    return nothing
end

"""
    _scvx__print_info(spbm[, err])

Print command line info message.

# Arguments
- `spbm`: the subproblem that was solved.
- `err`: an SCvx-specific error message.
"""
function _scvx__print_info(spbm::SCvxSubproblem,
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
        E = spbm.def.common.E
        status = @sprintf "%s" sol.status
        status = status[1:min(8, length(status))]
        ρ = !isnan(sol.ρ) ? @sprintf("%.2f", sol.ρ) : ""
        ρ = (length(ρ)>8) ? @sprintf("%.1e", sol.ρ) : ρ

        # Associate values with columns
        assoc = Dict(:iter => spbm.iter,
                     :status => status,
                     :maxvd => norm(sol.vd, Inf),
                     :maxvs => norm(sol.vs, Inf),
                     :maxvbc => norm([sol.vic; sol.vtc], Inf),
                     :cost => sol.J_aug,
                     :dx => max_dxh,
                     :du => max_duh,
                     :dp => max_dph,
                     :dynfeas => sol.feas ? "T" : "F",
                     :δ => sol.deviation,
                     :ρ => ρ,
                     :pre_improv => sol.pre_improv/abs(ref.J_aug)*100,
                     :dtr => sol.tr_update,
                     :rej => sol.reject ? "x" : "",
                     :tr => spbm.η)

        print(assoc, table)
    end

    _scp__overhead!(spbm)

    return nothing
end
