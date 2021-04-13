#= PTR algorithm data structures and methods.

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
include("../utils/helper.jl")
include("problem.jl")
include("scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Structure holding the PTR algorithm parameters. =#
struct PTRParameters <: SCPParameters
    N::T_Int          # Number of temporal grid nodes
    Nsub::T_Int       # Number of subinterval integration time nodes
    iter_max::T_Int   # Maximum number of iterations
    wvc::T_Real       # Virtual control weight
    wtr::T_Real       # Trust region weight
    ε_abs::T_Real     # Absolute convergence tolerance
    ε_rel::T_Real     # Relative convergence tolerance
    feas_tol::T_Real  # Dynamic feasibility tolerance
    q_tr::T_Real      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::T_Real    # Stopping criterion norm
    solver::Module    # The numerical solver to use for the subproblems
    solver_opts::Dict{T_String, Any} # Numerical solver options
end

#= PTR subproblem solution. =#
mutable struct PTRSubproblemSolution <: SCPSubproblemSolution
    iter::T_Int          # PTR iteration number
    # >> Discrete-time rajectory <<
    xd::T_RealMatrix     # States
    ud::T_RealMatrix     # Inputs
    p::T_RealVector      # Parameter vector
    # >> Virtual control terms <<
    vd::T_RealMatrix     # Dynamics virtual control
    vs::T_RealMatrix     # Nonconvex constraints virtual control
    vic::T_RealVector    # Initial conditions virtual control
    vtc::T_RealVector    # Terminal conditions virtual control
    # >> Cost values <<
    J::T_Real            # The original cost
    J_tr::T_Real         # The trust region penalty
    J_vc::T_Real         # The virtual control penalty
    J_aug::T_Real        # Overall cost
    # >> Trajectory properties <<
    ηx::T_RealVector     # State trust region radii
    ηu::T_RealVector     # Input trust region radii
    ηp::T_Real           # Parameter trust region radii
    status::T_ExitStatus # Numerical optimizer exit status
    feas::T_Bool         # Dynamic feasibility flag
    defect::T_RealMatrix # "Defect" linearization accuracy metric
    deviation::T_Real    # Deviation from reference trajectory
    unsafe::T_Bool       # Indicator that the solution is unsafe to use
    dyn::T_DLTV          # The dynamics
end

#= Subproblem definition in JuMP format for the convex numerical optimizer. =#
mutable struct PTRSubproblem <: SCPSubproblem
    iter::T_Int          # PTR iteration number
    mdl::Model           # The optimization problem handle
    algo::T_String       # SCP and convex algorithms used
    # >> Algorithm parameters <<
    def::SCPProblem      # The PTR problem definition
    # >> Reference and solution trajectories <<
    sol::Union{PTRSubproblemSolution, Missing} # Solution trajectory
    ref::Union{PTRSubproblemSolution, Missing} # Reference trajectory
    # >> Cost function <<
    J::T_Objective       # The original convex cost function
    J_tr::T_Objective    # The virtual control penalty
    J_vc::T_Objective    # The virtual control penalty
    J_aug::T_Objective   # Overall cost function
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
    # >> Trust region <<
    ηx::T_OptiVarVector  # State trust region radii
    ηu::T_OptiVarVector  # Input trust region radii
    ηp::T_OptiVar        # Parameter trust region radii
    # >> Statistics <<
    nvar::T_Int                    # Total number of decision variables
    ncons::Dict{T_Symbol, Any}     # Number of constraints
    timing::Dict{T_Symbol, T_Real} # Runtime profiling
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Construct the PTR problem definition.

Args:
    pars: PTR algorithm parameters.
    traj: the underlying trajectory optimization problem.

Returns:
    pbm: the problem structure ready for being solved by PTR. =#
function PTRProblem(pars::PTRParameters,
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
        (:trp, "ηp", "%.2f", 5)])

    pbm = SCPProblem(pars, traj, table)

    return pbm
end

#= Constructor for an empty convex optimization subproblem.

No cost or constraints. Just the decision variables and empty associated
parameters.

Args:
    pbm: the PTR problem being solved.
    iter: PTR iteration number.
    ref: (optional) the reference trajectory.

Returns:
    spbm: the subproblem structure. =#
function PTRSubproblem(pbm::SCPProblem,
                       iter::T_Int,
                       ref::Union{PTRSubproblemSolution,
                                  Missing}=missing)::PTRSubproblem

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
    algo = @sprintf("PTR (backend: %s)", cvx_algo)

    sol = missing # No solution associated yet with the subproblem

    # Cost
    J = missing
    J_tr = missing
    J_vc = missing
    J_aug = missing

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

    # Trust region radii
    ηx = @variable(mdl, [1:N], base_name="ηx")
    ηu = @variable(mdl, [1:N], base_name="ηu")
    ηp = @variable(mdl, base_name="ηp")

    spbm = PTRSubproblem(iter, mdl, algo, pbm, sol, ref, J, J_tr, J_vc,
                         J_aug, xh, uh, ph, x, u, p, vd, vs, vic, vtc,
                         ηx, ηu, ηp, nvar, ncons, timing)

    return spbm
end

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
function PTRSubproblemSolution(
    x::T_RealMatrix,
    u::T_RealMatrix,
    p::T_RealVector,
    iter::T_Int,
    pbm::SCPProblem)::PTRSubproblemSolution

    # Parameters
    N = pbm.pars.N
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    nv = size(pbm.common.E, 2)

    # Uninitialized parts
    ηx = fill(NaN, N)
    ηu = fill(NaN, N)
    ηp = NaN
    status = MOI.OPTIMIZE_NOT_CALLED
    feas = false
    defect = fill(NaN, nx, N-1)
    deviation = NaN
    unsafe = false
    dyn = T_DLTV(nx, nu, np, nv, N)

    vd = T_RealMatrix(undef, 0, N)
    vs = T_RealMatrix(undef, 0, N)
    vic = T_RealVector(undef, 0)
    vtc = T_RealVector(undef, 0)

    J = NaN
    J_tr = NaN
    J_vc = NaN
    J_aug = NaN
    J_aug = NaN

    subsol = PTRSubproblemSolution(iter, x, u, p, vd, vs, vic, vtc, J,
                                   J_tr, J_vc, J_aug, ηx, ηu, ηp, status,
                                   feas, defect, deviation, unsafe, dyn)

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
function PTRSubproblemSolution(spbm::PTRSubproblem)::PTRSubproblemSolution
    # Extract the discrete-time trajectory
    x = value.(spbm.x)
    u = value.(spbm.u)
    p = value.(spbm.p)

    # Form the partly uninitialized subproblem
    sol = PTRSubproblemSolution(x, u, p, spbm.iter, spbm.def)

    # Save the virtual control values and penalty terms
    sol.vd = value.(spbm.vd)
    sol.vs = value.(spbm.vs)
    sol.vic = value.(spbm.vic)
    sol.vtc = value.(spbm.vtc)

    # Save the optimal cost values
    sol.J = value(spbm.J)
    sol.J_tr = value(spbm.J_tr)
    sol.J_vc = value(spbm.J_vc)
    sol.J_aug = value(spbm.J_aug)

    # Save the solution status
    sol.ηx = value.(spbm.ηx)
    sol.ηu = value.(spbm.ηu)
    sol.ηp = value(spbm.ηp)

    return sol
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Apply the PTR algorithm to solve the trajectory generation problem.

Args:
    pbm: the trajectory problem to be solved.

Returns:
    sol: the PTR solution structure.
    history: PTR iteration data history. =#
function ptr_solve(pbm::SCPProblem)::Tuple{Union{SCPSolution, Nothing},
                                             SCPHistory}
    # ..:: Initialize ::..

    ref = _ptr__generate_initial_guess(pbm)

    history = SCPHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        spbm = PTRSubproblem(pbm, k, ref)

        _scp__add_dynamics!(spbm)
        _scp__add_convex_state_constraints!(spbm)
        _scp__add_convex_input_constraints!(spbm)
        _scp__add_nonconvex_constraints!(spbm)
        _scp__add_bcs!(spbm)
        _ptr__add_trust_region!(spbm)
        _ptr__add_cost!(spbm)

        _scp__save!(history, spbm)

        try
            # >> Solve the subproblem <<
            _scp__solve_subproblem!(spbm)

            # "Emergency exit" the PTR loop if something bad happened
            # (e.g. numerical problems)
            if _scp__unsafe_solution(spbm)
                _ptr__print_info(spbm)
                break
            end

            # >> Check stopping criterion <<
            stop = _ptr__check_stopping_criterion!(spbm)
            if stop
                _ptr__print_info(spbm)
                break
            end

            # >> Update reference trajectory <<
            ref = spbm.sol
        catch e
            isa(e, SCPError) || rethrow(e)
            _ptr__print_info(spbm, e)
            break
        end

        # >> Print iteration info <<
        _ptr__print_info(spbm)
    end

    reset(pbm.common.table)

    # ..:: Save solution ::..
    sol = SCPSolution(history)

    return sol, history
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the initial trajectory guess.

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to an PTRSubproblemSolution structure.

Args:
    pbm: the PTR problem structure.

Returns:
    guess: the initial guess. =#
function _ptr__generate_initial_guess(
    pbm::SCPProblem)::PTRSubproblemSolution

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    guess = PTRSubproblemSolution(x, u, p, 0, pbm)

    return guess
end

#= Add trust region constraint to the subproblem.

Args:
    spbm: the subproblem definition. =#
function _ptr__add_trust_region!(spbm::PTRSubproblem)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    q = spbm.def.pars.q_tr
    scale = spbm.def.common.scale
    traj_pbm = spbm.def.traj
    nx = traj_pbm.nx
    nu = traj_pbm.nu
    np = traj_pbm.np
    ηx = spbm.ηx
    ηu = spbm.ηu
    ηp = spbm.ηp
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

    # >> Trust region constraint <<
    q2cone = Dict(1 => :l1, 2 => :soc, 4 => :soc, Inf => :linf)
    cone = q2cone[q]
    C = T_ConvexConeConstraint
    acc! = add_conic_constraint!

    # Parameter trust region
    dp_lq = @variable(spbm.mdl, base_name="dp_lq")
    acc!(spbm.mdl, C(vcat(dp_lq, dp), cone))
    if q==4
        wp = @variable(spbm.mdl, base_name="wp")
        acc!(spbm.mdl, C(vcat(wp, dp_lq), :soc))
        acc!(spbm.mdl, C(vcat(wp, ηp, 1), :geom))
    else
        acc!(spbm.mdl, C(dp_lq-ηp, :nonpos))
    end

    # State and input trust regions
    dx_lq = @variable(spbm.mdl, [1:N], base_name="dx_lq")
    du_lq = @variable(spbm.mdl, [1:N], base_name="du_lq")
    for k = 1:N
        acc!(spbm.mdl, C(vcat(@k(dx_lq), @k(dx)), cone))
        acc!(spbm.mdl, C(vcat(@k(du_lq), @k(du)), cone))
        if q==4
            # State
            wx = @variable(spbm.mdl, base_name="wx")
            acc!(spbm.mdl, C(vcat(wx, @k(dx_lq)), :soc))
            acc!(spbm.mdl, C(vcat(wx, @k(ηx), 1), :geom))
            # Input
            wu = @variable(spbm.mdl, base_name="wu")
            acc!(spbm.mdl, C(vcat(wu, @k(du_lq)), :soc))
            acc!(spbm.mdl, C(vcat(wu, @k(ηu), 1), :geom))
        else
            # State
            acc!(spbm.mdl, C(@k(dx_lq)-@k(ηx), :nonpos))
            # Input
            acc!(spbm.mdl, C(@k(du_lq)-@k(ηu), :nonpos))
        end
    end

    return nothing
end

#= Define the subproblem cost function.

Args:
    spbm: the subproblem definition. =#
function _ptr__add_cost!(spbm::PTRSubproblem)::Nothing

    # Variables and parameters
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Compute the cost components
    spbm.J = _scp__original_cost(x, u, p, spbm.def)
    _ptr__compute_trust_region_penalty!(spbm)
    _ptr__compute_virtual_control_penalty!(spbm)

    # Overall cost
    spbm.J_aug = spbm.J+spbm.J_tr+spbm.J_vc

    # Associate cost function with the model
    set_objective_function(spbm.mdl, spbm.J_aug)
    set_objective_sense(spbm.mdl, MOI.MIN_SENSE)

    return nothing
end

#= Compute the subproblem cost trust region penalty term.

Args:
    spbm: the subproblem definition. =#
function _ptr__compute_trust_region_penalty!(spbm::PTRSubproblem)::Nothing

    # Variables and parameters
    t = spbm.def.common.t_grid
    wtr = spbm.def.pars.wtr
    ηx = spbm.ηx
    ηu = spbm.ηu
    ηp = spbm.ηp

    spbm.J_tr = wtr*(trapz(ηx, t)+trapz(ηu, t)+ηp)

    return nothing
end

#= Compute the subproblem cost virtual control penalty term.

Args:
    spbm: the subproblem definition. =#
function _ptr__compute_virtual_control_penalty!(spbm::PTRSubproblem)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    wvc = spbm.def.pars.wvc
    t = spbm.def.common.t_grid
    E = spbm.ref.dyn.E
    vd = spbm.vd
    vs = spbm.vs
    vic = spbm.vic
    vtc = spbm.vtc

    # Compute virtual control penalty
    C = T_ConvexConeConstraint
    acc! = add_conic_constraint!
    P = @variable(spbm.mdl, [1:N], base_name="P")
    Pf = @variable(spbm.mdl, [1:2], base_name="Pf")
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
    spbm.J_vc = wvc*(trapz(P, t)+sum(Pf))

    return nothing
end

#= Check if stopping criterion is triggered.

Args:
    spbm: the subproblem definition.

Returns:
    stop: true if stopping criterion holds. =#
function _ptr__check_stopping_criterion!(spbm::PTRSubproblem)::T_Bool

    # Extract values
    pbm = spbm.def
    ref = spbm.ref
    sol = spbm.sol
    ε_abs = pbm.pars.ε_abs
    ε_rel = pbm.pars.ε_rel

    # Compute solution deviation from reference
    sol.deviation = _scp__solution_deviation(spbm)

    # Check predicted cost improvement
    J_ref = ref.J_aug
    J_sol = sol.J_aug
    improv_rel = abs(J_ref-J_sol)/abs(J_ref)

    # Compute stopping criterion
    stop = (spbm.iter>1 &&
            (sol.feas && (improv_rel<=ε_rel || sol.deviation<=ε_abs)))

    return stop
end

#= Print command line info message.

Args:
    spbm: the subproblem that was solved.
    err: an PTR-specific error message. =#
function _ptr__print_info(spbm::PTRSubproblem,
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
        ΔJ = cost_improvement_percent(sol.J_aug, ref.J_aug)
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

        print(assoc, table)
    end

    _scp__overhead!(spbm)

    return nothing
end
