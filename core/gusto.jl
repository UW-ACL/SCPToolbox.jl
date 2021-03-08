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

include("../utils/types.jl")
include("problem.jl")
include("scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Structure holding the GuSTO algorithm parameters. =#
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
    ε::T_Real         # Convergence tolerance
    feas_tol::T_Real  # Dynamic feasibility tolerance
    pen::T_Symbol     # Penalty type (:quad, :softplus)
    hom::T_Real       # Homotopy parameter to use when pen==:softplus
    q_tr::T_Real      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::T_Real    # Stopping criterion norm
    solver::Module    # The numerical solver to use for the subproblems
    solver_opts::Dict{T_String, Any} # Numerical solver options
end

#= GuSTO subproblem solution. =#
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
    L::T_Real            # J *linearized* about reference solution
    L_st::T_Real         # J_st *linearized* about reference solution
    # >> Trajectory properties <<
    status::T_ExitStatus # Numerical optimizer exit status
    feas::T_Bool         # Dynamic feasibility flag
    defect::T_RealMatrix # "Defect" linearization accuracy metric
    deviation::T_Real    # Deviation from reference trajectory
    unsafe::T_Bool       # Indicator that the solution is unsafe to use
    cost_error::T_Real   # Cost error committed
    dyn_error::T_Real    # Cumulative dynamics error committed
    ρ::T_Real            # Convexification performance metric
    tr_update::T_String  # Indicator of growth direction for trust region
    reject::T_Bool       # Indicator whether GuSTO rejected this solution
    dyn::T_DLTV          # The dynamics
end
const T_GuSTOSubSol = GuSTOSubproblemSolution # Alias

#= Subproblem definition in JuMP format for the convex numerical optimizer. =#
mutable struct GuSTOSubproblem <: SCPSubproblem
    iter::T_Int                  # GuSTO iteration number
    mdl::Model                   # The optimization problem handle
    # >> Algorithm parameters <<
    def::SCPProblem              # The GuSTO algorithm definition
    λ::T_Real                    # Soft penalty weight
    η::T_Real                    # Trust region radius
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
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Construct the GuSTO problem definition.

Args:
    pars: GuSTO algorithm parameters.
    traj: the underlying trajectory optimization problem.

Returns:
    pbm: the problem structure ready for being solved by GuSTO. =#
function GuSTOProblem(pars::GuSTOParameters,
                      traj::TrajectoryProblem)::SCPProblem

    table = T_Table([
        # Iteration count
        (:iter, "k", "%d", 2),
        # Solver status
        (:status, "status", "%s", 8)])

    pbm = SCPProblem(pars, traj, table)

    return pbm
end

#= Constructor for an empty convex optimization subproblem.

No cost or constraints. Just the decision variables and empty associated
parameters.

Args:
    pbm: the GuSTO problem being solved.
    iter: GuSTO iteration number.
    λ: the soft penalty weight.
    η: the trust region radius.
    ref: the reference trajectory.

Returns:
    spbm: the subproblem structure. =#
function GuSTOSubproblem(pbm::SCPProblem,
                         iter::T_Int,
                         λ::T_Real,
                         η::T_Real,
                         ref::T_GuSTOSubSol)::GuSTOSubproblem

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

    spbm = GuSTOSubproblem(iter, mdl, pbm, λ, η, sol, ref, L, L_st, L_tr, L_vc,
                           L_aug, xh, uh, ph, x, u, p, vd)

    return spbm
end

#= Construct a subproblem solution from a discrete-time trajectory.

This leaves parameters of the solution other than the passed discrete-time
trajectory unset.

Args:
    x: discrete-time state trajectory.
    u: discrete-time input trajectory.
    p: parameter vector.
    iter: GuSTO iteration number.
    pbm: the GuSTO problem definition.

Returns:
    subsol: subproblem solution structure. =#
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
    reject = false
    dyn = T_DLTV(nx, nu, np, nv, N)

    vd = T_RealMatrix(undef, 0, N)

    J = NaN
    J_st = NaN
    J_tr = NaN
    J_vc = NaN
    L = NaN
    L_st = NaN

    subsol = GuSTOSubproblemSolution(iter, x, u, p, vd, J, J_st, J_tr, J_vc, L,
                                     L_st, status, feas, defect, deviation,
                                     unsafe, cost_error, dyn_error, ρ,
                                     tr_update, reject, dyn)

    # Compute the DLTV dynamics around this solution
    _scp__discretize!(subsol, pbm)

    return subsol
end

#= Construct subproblem solution from a subproblem object.

Expects that the subproblem argument is a solved subproblem (i.e. one to which
numerical optimization has been applied).

Args:
    spbm: the subproblem structure.

Returns:
    sol: subproblem solution. =#
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
    sol.L = value.(spbm.L)
    sol.J_vc = value.(spbm.L_vc)
    # TODO

    # Save the solution status
    sol.status = termination_status(spbm.mdl)

    return sol
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Apply the GuSTO algorithm to solve the trajectory generation problem.

Args:
    pbm: the trajectory problem to be solved.

Returns:
    sol: the GuSTO solution structure.
    history: GuSTO iteration data history. =#
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

        _scp__add_dynamics!(spbm)
        _gusto__add_cost!(spbm)
        # TODO _gusto__add_<...>!(spbm)

        _scp__save!(history, spbm)

        try
            # >> Solve the subproblem <<
            _scp__solve_subproblem!(spbm)

            # TODO
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
    sol = ref # TODO SCPSolution(history)

    return sol, history
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the initial trajectory guess.

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to a GuSTOSubproblemSolution structure.

Args:
    pbm: the GuSTO problem structure.

Returns:
    guess: the initial guess. =#
function _gusto__generate_initial_guess(
    pbm::SCPProblem)::T_GuSTOSubSol

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    guess = T_GuSTOSubSol(x, u, p, 0, pbm)

    return guess
end

#= Define the subproblem cost function.

Args:
    spbm: the subproblem definition. =#
function _gusto__add_cost!(spbm::GuSTOSubproblem)::Nothing

    # Variables and parameters
    x = spbm.x
    u = spbm.u
    p = spbm.p
    vd = spbm.vd

    # Compute the cost components
    spbm.L = _gusto__original_cost(x, u, p, spbm)
    spbm.L_st = 0.0 # TODO _gusto__state_penalty_cost(x, u, p, spbm)
    spbm.L_tr = _gusto__trust_region_cost(x, p, spbm)
    spbm.L_vc = _gusto__virtual_control_cost(vd, spbm)

    # Overall cost
    spbm.L_aug = spbm.L+spbm.L_st+spbm.L_tr+spbm.L_vc

    # Associate cost function with the model
    set_objective_function(spbm.mdl, spbm.L_aug)
    set_objective_sense(spbm.mdl, MOI.MIN_SENSE)

    return nothing
end

#= Compute the original cost function.

This function has two "modes": the (default) convex mode computes the convex
version of the cost (where all non-convexity has been convexified), while the
nonconvex mode computes the fully nonlinear cost.

Args:
    x: the discrete-time state trajectory.
    u: the discrete-time input trajectory.
    p: the parameter vector.
    spbm: the subproblem structure.
    mode: (optional) either :convex (default) or :nonconvex.

Returns:
    cost: the original cost. =#
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
    τ_grid = spbm.def.common.τ_grid
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
            if !isnothing(traj.S)
                if traj.S_cvx
                    Γk += @k(u)'*traj.S(p)*@k(u)
                else
                    S = traj.S(pb)
                    ∇S = traj.dSdp(pb)
                    dp = p-pb
                    S1 = S+∇S.*dp
                    Γk += @k(u)'*S1*@k(u)
                end
            end
            if !isnothing(traj.ℓ)
                if traj.ℓ_cvx
                    Γk += @k(u)'*traj.ℓ(@k(x), p)
                else
                    uℓ = @k(ub)'*traj.ℓ(@k(xb), pb)
                    ∇uuℓ = traj.ℓ(@k(xb), pb)
                    ∇xuℓ = traj.dℓdx(@k(xb), pb)'*@k(ub)
                    ∇puℓ = traj.dℓdp(@k(xb), pb)'*@k(ub)
                    du = @k(u)-@k(ub)
                    dx = @k(x)-@k(xb)
                    dp = p-pb
                    uℓ1 = uℓ+∇uuℓ'*du+∇xuℓ'*dx+∇puℓ'*dp
                    Γk += uℓ1
                end
            end
            if !isnothing(traj.g)
                if traj.g_cvx
                    Γk += traj.g(@k(x), p)
                else
                    g = traj.g(@k(xb), pb)
                    ∇xg = traj.dgdx(@k(xb), pb)
                    ∇pg = traj.dgdp(@k(xb), pb)
                    dx = @k(x)-@k(xb)
                    dp = p-pb
                    g1 = g+∇xg'*dx+∇pg'*dp
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
    cost_run = trapz(cost_run_integrand, τ_grid)

    # Overall original cost
    cost = cost_term+cost_run

    return cost
end

#= Compute a smooth, convex and nondecreasing penalization function.

If the Jacobian values are passed in, a linearized version of the penalty
function is computed. If one of the Jacobians is zero (e.g. d/dx or d/dp), then
you can pass `nothing` in its place.

Args:
    spbm: the subproblem structure.
    f: the quantity to be penalized.
    dfdx: (optional) Jacobian of f wrt state.
    dfdp: (optional) Jacobian of f wrt parameter vector.
    dx: (optional) state vector deviation from reference.
    dp: (optional) parameter vector deviation from reference.

Returns:
    h: penalization function value. =#
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
    if linearized
        dfdx = !isnothing(dfdx) ? dfdx : zeros(traj.nx)
        dfdp = !isnothing(dfdp) ? dfdp : zeros(traj.np)
        dx = !isnothing(dx) ? dx : zeros(traj.nx)
        dp = !isnothing(dp) ? dp : zeros(traj.np)
    end

    # Compute the function value
    # The possibilities are:
    #   (:quad)      h(f(x, p)) = λ*(max(0, f(x, p)))^2
    #   (:softplus)  h(f(x, p)) = λ*log(1+exp(hom*f(x, p)))/hom
    if penalty==:quad
        # ..:: Quadratic penalty ::..
        if mode==:numerical || (mode==:jump && linearized)
            h = (max(0.0, f))^2
            if linearized
                if f<0
                    h = 0.0
                else
                    dhdx, dhdp = 2*f*dfdx, 2*f*dfdp
                    h = h+dhdx*dx+dhdp*dp
                end
            end
        else
            u = @variable(spbm.mdl, base_name="u")
            v = @variable(spbm.mdl, base_name="v")
            @constraint(spbm.mdl, u<=0.0)
            @constraint(spbm.mdl, f-u <= v)
            @constraint(spbm.mdl, -v <= f-u)
            h = v^2
        end
    else
        # ..:: Log-sum-exp penalty ::..
        if mode==:numerical || (mode==:jump && linearized)
            F = [0, f]
            h = logsumexp(F; t=hom)
            if linearized
                dFdx = [zeros(traj.nx), dfdx]
                dFdp = [zeros(traj.np), dfdp]
                _, dhdx = logsumexp(F, dFdx; t=hom)
                _, dhdp = logsumexp(F, dFdp; t=hom)
                h = h+dhdx*dx+dhdp*dp
            end
        else
            u = @variable(spbm.mdl, base_name="u")
            v = @variable(spbm.mdl, base_name="v")
            w = @variable(spbm.mdl, base_name="w")
            acc! = add_conic_constraint!
            C = T_ConvexConeConstraint
            acc!(spbm.mdl, C(vcat(-w, 1, u), :exp))
            acc!(spbm.mdl, C(vcat(hom*f-w, 1, v), :exp))
            @constraint(spbm.mdl, u+v <= 1)
            h = w/hom
        end
    end
    h *= λ

    return h
end

#= Compute the trust region constraint soft penalty.

This function has two "modes": the (default) convex mode computes the convex
version of the cost (where all non-convexity has been convexified), while the
nonconvex mode computes the fully nonlinear cost.

Args:
    x: the discrete-time state trajectory.
    u: the discrete-time input trajectory.
    p: the parameter vector.
    spbm: the subproblem structure.
    mode: (optional) either :convex (default) or :nonconvex.

Returns:
    cost_tr: the trust region soft penalty cost. =#
function _gusto__trust_region_cost(x::T_OptiVarMatrix,
                                   p::T_OptiVarVector,
                                   spbm::GuSTOSubproblem,
                                   mode::T_Symbol=:convex)::T_Objective

    # Parameters
    pars = spbm.def.pars
    scale = spbm.def.common.scale
    q = pars.q_tr
    N = pars.N
    η = spbm.η
    sqrt_η = sqrt(η)
    τ_grid = spbm.def.common.τ_grid
    xh = scale.iSx*(x.-scale.cx)
    ph = scale.iSp*(p-scale.cp)
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)
    dx = xh-xh_ref
    dp = ph-ph_ref
    if mode==:convex
        tr = @variable(spbm.mdl, [1:N], base_name="tr")
    end

    # Integrated running cost
    cost_tr_integrand = Vector{T_Objective}(undef, N)
    C = T_ConvexConeConstraint
    acc! = add_conic_constraint!
    if q==2
        dx_l2 = @variable(spbm.mdl, [1:N], base_name="dx_l2")
        dp_l2 = @variable(spbm.mdl, base_name="dp_l2")
        acc!(spbm.mdl, C(vcat(dp_l2, dp), :soc))
    end
    for k = 1:N
        if mode==:convex
            tr_k = @k(tr)
            if q==1
                # 1-norm
                tr_cone = C(vcat(η+tr_k, @k(dx), dp), :l1)
                add_conic_constraint!(spbm.mdl, tr_cone)
            elseif q==2
                # 2-norm
                acc!(spbm.mdl, C(vcat(@k(dx_l2), @k(dx)), :soc))
                @constraint(spbm.mdl, @k(dx_l2)+dp_l2 <= η+tr_k)
            elseif q==4
                # 2-norm squared
                tr_cone = C(vcat(sqrt_η+tr_k, @k(dx), dp), :soc)
                add_conic_constraint!(spbm.mdl, tr_cone)
            else
                # Infinity-norm
                tr_cone = C(vcat(η+tr_k, @k(dx), dp), :linf)
                add_conic_constraint!(spbm.mdl, tr_cone)
            end
        else
            if q==4
                tr_k = @k(dx)'*@k(dx)+@k(dp)'*@k(dp)-η
            else
                tr_k = norm(@k(dx), q)+norm(@k(dp), q)-η
            end
        end
        @k(cost_tr_integrand) = _gusto__soft_penalty(spbm, tr_k)
    end
    cost_tr = trapz(cost_tr_integrand, τ_grid)

    return cost_tr
end

#= Compute the virtual control penalty.

This function has two "modes": the (default) convex mode computes the convex
version of the cost (where all non-convexity has been convexified), while the
nonconvex mode computes the fully nonlinear cost.

Args:
    vd: the discrete-time dynamics virtual control trajectory.
    spbm: the subproblem structure.
    mode: (optional) either :convex (default) or :nonconvex.

Returns:
    cost_vc: the virtual control penalty cost. =#
function _gusto__virtual_control_cost(vd::T_OptiVarMatrix,
                                      spbm::GuSTOSubproblem,
                                      mode::T_Symbol=:convex)::T_Objective

    # Parameters
    pars = spbm.def.pars
    ω = pars.ω
    N = pars.N
    τ_grid = spbm.def.common.τ_grid
    if mode==:convex
        E = spbm.ref.dyn.E
        vc_l1 = @variable(spbm.mdl, [1:N-1], base_name="vc_l1")
    end

    # Integrated running cost
    cost_vc_integrand = Vector{T_Objective}(undef, N)
    for k = 1:N
        if mode==:convex
            # Evaluation using virtual control in an optimization subproblem
            if k<N
                vck_l1 = @k(vc_l1)
                C = T_ConvexConeConstraint(vcat(vck_l1, @k(E)*@k(vd)), :l1)
                add_conic_constraint!(spbm.mdl, C)
            else
                vck_l1 = 0.0
            end
        else
            # Evaluation using nonlinear propagation defects
            vck_l1 = norm(@k(vd), 1)
        end
        @k(cost_vc_integrand) = ω*vck_l1
    end
    cost_vc = trapz(cost_vc_integrand, τ_grid)

    return cost_vc
end

#= Print command line info message.

Args:
    spbm: the subproblem that was solved.
    err: a GuSTO-specific error message. =#
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
        status = @sprintf "%s" sol.status
        status = status[1:min(8, length(status))]

        # Associate values with columns
        assoc = Dict(:iter => spbm.iter,
                     :status => status)

        print(assoc, table)
    end

    return nothing
end
