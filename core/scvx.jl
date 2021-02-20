#= This file stores the data structures and methods for the SCvx algorithm. =#

using LinearAlgebra
using JuMP
using ECOS
using Printf

include("../utils/types.jl")
include("../utils/helper.jl")
include("problem.jl")

# ..:: Data structures ::..

#= Structure holding the SCvx algorithm parameters. =#
struct SCvxParameters
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

#= Variable scaling parameters.

Holds the SCvx internal scaling parameters, which makes the numerical
optimization subproblems better conditioned. =#
struct SCvxScaling
    Sx::T_RealMatrix  # State scaling coefficient matrix
    cx::T_RealVector  # State scaling offset vector
    Su::T_RealMatrix  # Input scaling coefficient matrix
    cu::T_RealVector  # Input scaling offset vector
    Sp::T_RealMatrix  # Parameter scaling coefficient matrix
    cp::T_RealVector  # Parameter scaling offset matrix
    iSx::T_RealMatrix # Inverse of state scaling matrix
    iSu::T_RealMatrix # Inverse of input scaling matrix
    iSp::T_RealMatrix # Inverse of parameter scaling coefficient matrix
end

#= Indexing arrays for convenient access during dynamics discretization.

Container of indices useful for extracting variables from the propagation
vector during the linearized dynamics discretization process. =#
struct SCvxDiscretizationIndices
    x::T_IntRange  # Indices for state
    A::T_IntRange  # Indices for A matrix
    Bm::T_IntRange # Indices for B_{-} matrix
    Bp::T_IntRange # Indices for B_{+} matrix
    F::T_IntRange  # Indices for S matrix
    r::T_IntRange  # Indices for r vector
    E::T_IntRange  # Indices for E matrix
    length::T_Int  # Propagation vector total length
end

#= Subproblem solution.

Structure which stores a solution obtained by solving the subproblem during an
SCvx iteration. =#
mutable struct SCvxSubproblemSolution
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
    L_orig::T_Real       # The original convex cost function
    L_pen::T_Real        # The virtual control penalty
    L::T_Real            # Overall linear cost
    J::T_Real            # Overall nonlinear cost
    dJ::T_Real           # Actual cost improvement
    dL::T_Real           # Predicted cost improvement
    # >> Trajectory properties <<
    status::T_ExitStatus # Numerical optimizer exit status
    feas::T_Bool         # Dynamic feasibility flag
    defect::T_RealMatrix # "Defect" linearization accuracy metric
    deviation::T_Real    # Deviation from reference trajectory
    unsafe::T_Bool       # Indicator that the solution is unsafe to use
    ρ::T_Real            # Convexification performance metric
    tr_update::T_String  # Indicator of growth direction for trust region
    reject::T_Bool       # Indicator whether SCvx rejected this solution
    # >> Discrete-time dynamics update matrices <<
    # x[:, k+1] = ...
    A::T_RealTensor      # ...  A[:, :, k]*x[:, k]+ ...
    Bm::T_RealTensor     # ... +Bm[:, :, k]*u[:, k]+ ...
    Bp::T_RealTensor     # ... +Bp[:, :, k]*u[:, k+1]+ ...
    F::T_RealTensor      # ... +F[:, :, k]*p+ ...
    r::T_RealMatrix      # ... +r[:, k]+ ...
    E::T_RealTensor      # ... +E[:, :, k]*v
end

#= Common constant terms used throughout the algorithm. =#
struct SCvxCommon
    # >> Discrete-time grid <<
    Δτ::T_Real           # Discrete time step
    τ_grid::T_RealVector # Grid of scaled timed on the [0,1] interval
    # >> Virtual control <<
    E::T_RealMatrix      # Continuous-time matrix for dynamics virtual control
    # >> Scaling <<
    scale::SCvxScaling   # Variable scaling
    # >> Iteration info <<
    table::Table         # Iteration info table (printout to REPL)
end

#=Structure which contains all the necessary information to run SCvx.=#
struct SCvxProblem
    pars::SCvxParameters    # Algorithm parameters
    traj::TrajectoryProblem # The underlying trajectory problem
    common::SCvxCommon      # Common precomputed terms
end

#= Subproblem data in JuMP format for the convex numerical optimizer.

This structure holds the final data that goes into the subproblem
optimization. =#
mutable struct SCvxSubproblem
    iter::T_Int                  # SCvx iteration number
    mdl::Model                   # The optimization problem handle
    # >> Algorithm parameters <<
    scvx::SCvxProblem            # The SCvx problem definition
    # >> Reference and solution trajectories <<
    sol::Union{SCvxSubproblemSolution, Missing} # Solution trajectory
    ref::Union{SCvxSubproblemSolution, Missing} # Reference trajectory
    # >> Cost function <<
    L_orig::T_Objective          # The original convex cost function
    L_pen::T_Objective           # The virtual control penalty
    L::T_Objective               # Overall cost function
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
    vs::T_OptiVarMatrix          # Nonconvex constraints virtual control
    vic::T_OptiVarVector         # Initial conditions virtual control
    vtc::T_OptiVarVector         # Terminal conditions virtual control
    # >> Other variables <<
    P::T_OptiVarVector           # Virtual control penalty
    Pf::T_OptiVarVector          # Boundary condition virtual control penalty
    tr_rx::T_OptiVarMatrix       # 2-norm trust region JuMP formulation
    # >> Trust region <<
    η::T_Real                    # Trust region radius
    # >> Discrete-time dynamics update matrices <<
    # x[:,k+1] = ...
    A::T_RealTensor              # ...  A[:, :, k]*x[:, k]+ ...
    Bm::T_RealTensor             # ... +Bm[:, :, k]*u[:, k]+ ...
    Bp::T_RealTensor             # ... +Bp[:, :, k]*u[:, k+1]+ ...
    F::T_RealTensor              # ... +F[:, :, k]*p+ ...
    r::T_RealMatrix              # ... +r[:, k]+ ...
    E::T_RealTensor              # ... +E[:, :, k]*v
    # >> Constraint references <<
    dynamics::T_ConstraintMatrix # Dynamics
    ic::T_ConstraintVector       # Initial conditions
    tc::T_ConstraintVector       # Terminal conditions
    p_bnds::T_ConstraintVector   # Parameter bounds
    pc_x::T_ConstraintMatrix     # Path box constraints on state
    pc_u::T_ConstraintMatrix     # Path box constraints on input
    pc_X::T_ConstraintMatrix     # State convex path constraints
    pc_U::T_ConstraintMatrix     # Input convex path constraints
    pc_s::T_ConstraintMatrix     # Nonconvex path constraints
    tr_xu::T_ConstraintVector    # Trust region constraint
    fit::T_ConstraintVector      # Constraints to fit problem and JuMP template
end

#= Overall trajectory solution.

Structure which holds the trajectory solution that the SCvx algorithm
returns. =#
struct SCvxSolution
    # >> Properties <<
    status::T_String  # Solution status (success? failure?)
    iterations::T_Int # Number of SCvx iterations that occurred
    # >> Discrete-time trajectory <<
    τd::T_RealVector   # Discrete times
    xd::T_RealMatrix  # States
    ud::T_RealMatrix  # Inputs
    p::T_RealVector   # Parameter vector
    # >> Continuous-time trajectory <<
    xc::Union{ContinuousTimeTrajectory, Missing} # States
    uc::Union{ContinuousTimeTrajectory, Missing} # Inputs
    # >> Cost function value <<
    L_orig::T_Real    # The original convex cost function
    L_pen::T_Real     # The virtual control penalty
    L::T_Real         # Overall linear cost function of the subproblem
    J::T_Real         # Overall nonlinear cost function
end

#= SCvx iteration history.

Holds the history of SCvx iteration data. =#
struct SCvxHistory
    subproblems::Vector{SCvxSubproblem} # Subproblems
end

# ..:: Constructors ::..

#= Indexing arrays from problem definition.

Args:
    pbm: the SCvx problem definition.

Returns:
    idcs: the indexing array structure. =#
function SCvxDiscretizationIndices(pbm::SCvxProblem)::SCvxDiscretizationIndices
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    id_x  = (1:nx)
    id_A  = id_x[end].+(1:nx*nx)
    id_Bm = id_A[end].+(1:nx*nu)
    id_Bp = id_Bm[end].+(1:nx*nu)
    id_S  = id_Bp[end].+(1:nx*np)
    id_r  = id_S[end].+(1:nx)
    id_E  = id_r[end].+(1:length(pbm.common.E))
    id_sz = length([id_x; id_A; id_Bm; id_Bp; id_S; id_r; id_E])
    idcs = SCvxDiscretizationIndices(id_x, id_A, id_Bm, id_Bp, id_S,
                                     id_r, id_E, id_sz)
    return idcs
end

#= Construct a subproblem solution from a discrete-time trajectory.

This leaves parameters of the solution other than the passed discrete-time
trajectory unset.

Args:
    x: discrete-time state trajectory.
    u: discrete-time input trajectory.
    p: parameter vector.
    iter: SCvx iteration number.
    pbm: the SCvx problem definition.

Returns:
    subsol: subproblem solution structure. =#
function SCvxSubproblemSolution(
    x::T_RealMatrix,
    u::T_RealMatrix,
    p::T_RealVector,
    iter::T_Int,
    pbm::SCvxProblem)::SCvxSubproblemSolution

    # Parameters
    N = pbm.pars.N
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    _E = pbm.common.E

    # Uninitialized parts
    status = MOI.OPTIMIZE_NOT_CALLED
    feas = false
    defect = fill(NaN, nx, N-1)
    deviation = NaN
    unsafe = false
    ρ = NaN
    tr_update = ""
    reject = false

    vd = T_RealMatrix(undef, 0, N)
    vs = T_RealMatrix(undef, 0, N)
    vic = T_RealVector(undef, 0)
    vtc = T_RealVector(undef, 0)
    P = zeros(N)
    Pf = zeros(2)

    L_orig = NaN
    L_pen = NaN
    L = NaN
    J = NaN
    dJ = NaN
    dL = NaN

    A = T_RealTensor(undef, nx, nx, N-1)
    Bm = T_RealTensor(undef, nx, nu, N-1)
    Bp = T_RealTensor(undef, nx, nu, N-1)
    S = T_RealTensor(undef, nx, np, N-1)
    r = T_RealMatrix(undef, nx, N-1)
    E = T_RealTensor(undef, size(_E)..., N-1)

    subsol = SCvxSubproblemSolution(iter, x, u, p, vd, vs, vic, vtc, P, Pf,
                                    L_orig, L_pen, L, J, dJ, dL, status,
                                    feas, defect, deviation, unsafe, ρ,
                                    tr_update, reject, A, Bm, Bp, S, r, E)

    return subsol
end

#= Construct the SCvx problem definition.

This internally also computes the scaling matrices used to improve subproblem
numerics.

Args:
    pars: SCvx algorithm parameters.
    traj: the underlying trajectory optimization problem.

Returns:
    pbm: the problem structure ready for being solved by SCvx. =#
function SCvxProblem(pars::SCvxParameters,
                     traj::TrajectoryProblem)::SCvxProblem

    # Compute the common constant terms
    τ_grid = LinRange(0.0, 1.0, pars.N)
    Δτ = τ_grid[2]-τ_grid[1]
    E = I(traj.nx)
    scale = _scvx__compute_scaling(pars, traj)

    table = Table([
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
        # Original cost value
        (:cost, "J", "%.2e", 8),
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
        (:ρ, "ρ", "%s", 8),
        # Predicted cost improvement (percent)
        (:dL, "dL %", "%.2f", 8),
        # Update direction for trust region radius (grow? shrink?)
        (:dtr, "Δη", "%s", 3),
        # Reject solution indicator
        (:rej, "rej", "%s", 5)])

    consts = SCvxCommon(Δτ, τ_grid, E, scale, table)

    pbm = SCvxProblem(pars, traj, consts)

    return pbm
end

#= Construct subproblem solution from a subproblem object.

Expects that the subproblem argument is a solved subproblem (i.e. one to which
numerical optimization has been applied).

Args:
    spbm: the subproblem structure.

Returns:
    sol: subproblem solution. =#
function SCvxSubproblemSolution(spbm::SCvxSubproblem)::SCvxSubproblemSolution
    # Extract the discrete-time trajectory
    x = value.(spbm.x)
    u = value.(spbm.u)
    p = value.(spbm.p)

    # Form the partly uninitialized subproblem
    sol = SCvxSubproblemSolution(x, u, p, spbm.iter, spbm.scvx)

    # Save the virtual control values and penalty terms
    sol.vd = value.(spbm.vd)
    sol.vs = value.(spbm.vs)
    sol.vic = value.(spbm.vic)
    sol.vtc = value.(spbm.vtc)
    sol.P = value.(spbm.P)
    sol.Pf = value.(spbm.Pf)

    # Save the optimal cost values
    sol.L_orig = value(spbm.L_orig)
    sol.L_pen = value(spbm.L_pen)
    sol.L = value(spbm.L)
    _scvx__discretize!(sol, spbm.scvx)
    _scvx__solution_cost!(sol, :nonlinear, spbm.scvx)

    # Save the solution status
    sol.status = termination_status(spbm.mdl)

    return sol
end

#= Constructor for an empty convex optimization subproblem.

No cost or constraints. Just the decision variables and empty associated
parameters.

Args:
    pbm: the SCvx problem being solved.
    iter: SCvx iteration number.
    η: the trust region radius.
    ref: (optional) the reference trajectory.

Returns:
    spbm: the subproblem structure. =#
function SCvxSubproblem(pbm::SCvxProblem,
                        iter::T_Int,
                        η::T_Real,
                        ref::Union{SCvxSubproblemSolution,
                                   Missing}=missing)::SCvxSubproblem
    # Sizes
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    N = pbm.pars.N
    _E = pbm.common.E

    # Optimization problem handle
    solver = pbm.pars.solver
    solver_opts = pbm.pars.solver_opts
    mdl = Model()
    set_optimizer(mdl, solver.Optimizer)
    for (key,val) in solver_opts
        set_optimizer_attribute(mdl, key, val)
    end

    sol = missing # No solution associated yet with the subproblem

    # Cost (default: feasibility problem)
    L_orig = missing
    L_pen = missing
    L = missing

    # Decision variables (scaled)
    xh = @variable(mdl, [1:nx, 1:N], base_name="xh")
    uh = @variable(mdl, [1:nu, 1:N], base_name="uh")
    ph = @variable(mdl, [1:np], base_name="ph")

    # Physical decision variables
    x = pbm.common.scale.Sx*xh.+pbm.common.scale.cx
    u = pbm.common.scale.Su*uh.+pbm.common.scale.cu
    p = pbm.common.scale.Sp*ph.+pbm.common.scale.cp
    vd = @variable(mdl, [1:size(_E, 2), 1:N-1], base_name="vd")
    vs = T_RealMatrix(undef, 0, N)
    vic = T_RealVector(undef, 0)
    vtc = T_RealVector(undef, 0)

    # Other variables
    P = @variable(mdl, [1:N], base_name="P")
    Pf = @variable(mdl, [1:2], base_name="Pf")
    tr_rx = @variable(mdl, [1:2, 1:N], base_name="tr_xu")

    # Uninitialized parameters
    A = T_RealTensor(undef, nx, nx, N-1)
    Bm = T_RealTensor(undef, nx, nu, N-1)
    Bp = T_RealTensor(undef, nx, nu, N-1)
    F = T_RealTensor(undef, nx, np, N-1)
    r = T_RealMatrix(undef, nx, N-1)
    E = T_RealTensor(undef, size(_E)..., N-1)

    # Empty constraints
    dynamics = T_ConstraintMatrix(undef, nx, N-1)
    ic = T_ConstraintVector(undef, 0)
    tc = T_ConstraintVector(undef, 0)
    p_bnds = T_ConstraintVector(undef, np)
    pc_x = T_ConstraintMatrix(undef, nx, N)
    pc_u = T_ConstraintMatrix(undef, nu, N)
    pc_X = T_ConstraintMatrix(undef, 0, N)
    pc_U = T_ConstraintMatrix(undef, 0, N)
    pc_s = T_ConstraintMatrix(undef, 0, N)
    tr_xu = T_ConstraintVector(undef, N)
    fit = T_ConstraintVector(undef, 0)

    spbm = SCvxSubproblem(iter, mdl, pbm, sol, ref, L_orig, L_pen, L, xh, uh,
                          ph, x, u, p, vd, vs, vic, vtc, P, Pf, tr_rx, η, A,
                          Bm, Bp, F, r, E, dynamics, ic, tc, p_bnds, pc_x,
                          pc_u, pc_X, pc_U, pc_s, tr_xu, fit)

    return spbm
end

#= Convert subproblem solution to a final trajectory solution.

This is what the SCvx algorithm returns in the end to the user.

Args:
    history: SCvx iteration history.

Returns:
    sol: the trajectory solution. =#
function SCvxSolution(history::SCvxHistory)::SCvxSolution

    # Get the solution
    last_spbm = history.subproblems[end]
    last_sol = last_spbm.sol

    # Extract relevant parameters
    num_iters = last_spbm.iter
    pbm = last_spbm.scvx
    N = pbm.pars.N
    Nsub = pbm.pars.Nsub
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    τd = pbm.common.τ_grid

    if _scvx__unsafe_solution(last_sol)
        # SCvx failed :(
        status = @sprintf "%s (%s)" SCVX_FAILED last_sol.status
        xd = T_RealMatrix(undef, size(last_sol.xd))
        ud = T_RealMatrix(undef, size(last_sol.ud))
        p = T_RealVector(undef, size(last_sol.p))
        xc = missing
        uc = missing
        L_orig = Inf
        L_pen = Inf
        L = Inf
        J = Inf
    else
        # SCvx solved the problem!
        status = @sprintf "%s" SCVX_SOLVED

        xd = last_sol.xd
        ud = last_sol.ud
        p = last_sol.p

        # >> Continuos-time trajectory <<
        # Since within-interval integration using Nsub points worked, using
        # twice as many this time around seems like a good heuristic
        Nc = 2*Nsub*(N-1)
        τc = T_RealVector(LinRange(0.0, 1.0, Nc))
        uc = ContinuousTimeTrajectory(τd, ud, :linear)
        F = (τ, x) -> pbm.traj.f(τ, x, sample(uc, τ), p)
        xc_vals = rk4(F, @first(last_sol.xd), τc; full=true)
        xc = ContinuousTimeTrajectory(τc, xc_vals, :linear)

        L_orig = last_sol.L_orig
        L_pen = last_sol.L_pen
        L = last_sol.L
        J = last_sol.J
    end

    sol = SCvxSolution(status, num_iters, τd, xd, ud, p, xc, uc,
                       L_orig, L_pen, L, J)

    return sol
end

#= Empty history.

Returns:
    history: history with no entries. =#
function SCvxHistory()::SCvxHistory
    subproblems = Vector{SCvxSubproblem}(undef, 0)
    history = SCvxHistory(subproblems)
    return history
end

# ..:: Public methods ::..

#= Apply the SCvx algorithm to solve the trajectory generation problem.

Args:
    pbm: the trajectory problem to be solved.

Returns:
    sol: the SCvx solution structure.
    history: SCvx iteration data history. =#
function scvx_solve(pbm::SCvxProblem)::Tuple{Union{SCvxSolution, Nothing},
                                             SCvxHistory}
    # ..:: Initialize ::..

    η = pbm.pars.η_init
    ref = _scvx__generate_initial_guess(pbm)

    history = SCvxHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        spbm = SCvxSubproblem(pbm, k, η, ref)

        _scvx__add_dynamics!(spbm)
        _scvx__add_bcs!(spbm)
        _scvx__add_convex_constraints!(spbm)
        _scvx__add_nonconvex_constraints!(spbm)
        _scvx__add_trust_region!(spbm)
        _scvx__add_cost!(spbm)

        _scvx__save!(history, spbm)

        try
            # >> Solve the subproblem <<
            _scvx__solve_subproblem!(spbm)

            # "Emergency exit" the SCvx loop if something bad happened
            # (e.g. numerical problems)
            if _scvx__unsafe_solution(spbm)
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
            isa(e, SCvxError) || rethrow(e)
            _scvx__print_info(spbm, e)
            break
        end

        # >> Print iteration info <<
        _scvx__print_info(spbm)
    end

    # ..:: Save solution ::..
    sol = SCvxSolution(history)

    return sol, history
end

# ..:: Private methods ::..

#= Compute the initial trajectory guess.

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to an SCvxSubproblemSolution structure.

Args:
    pbm: the SCvx problem structure.

Returns:
    guess: the initial guess. =#
function _scvx__generate_initial_guess(
    pbm::SCvxProblem)::SCvxSubproblemSolution

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    _scvx__correct_convex!(x, u, pbm)
    guess = SCvxSubproblemSolution(x, u, p, 0, pbm)

    # Compute the nonlinear cost
    _scvx__discretize!(guess, pbm)
    _scvx__solution_cost!(guess, :nonlinear, pbm)

    return guess
end

#= Discrete linear time varying dynamics computation.

Compute the discrete-time update matrices for the linearized dynamics about a
reference trajectory. As a byproduct, this calculates the defects needed for
the trust region update.

Args:
    ref: the reference trajectory for which the propagation is done.
    pbm: the SCvx problem definition. =#
function _scvx__discretize!(ref::SCvxSubproblemSolution,
                            pbm::SCvxProblem)::Nothing
    # Parameters
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    N = pbm.pars.N
    Nsub = pbm.pars.Nsub
    _E = pbm.common.E

    # Initialization
    idcs = SCvxDiscretizationIndices(pbm)
    V0 = zeros(idcs.length)
    V0[idcs.A] = vec(I(nx))
    ref.feas = true

    # Propagate individually over each discrete-time interval
    for k = 1:N-1
        # Reset the state initial condition
        V0[idcs.x] = @k(ref.xd)

        # Integrate
        f = (τ::T_Real, V::T_RealVector) ->
            _scvx__derivs(τ, V, k, pbm, idcs, ref)
        τ_subgrid = T_RealVector(
            LinRange(pbm.common.τ_grid[k], pbm.common.τ_grid[k+1], Nsub))
        V = rk4(f, V0, τ_subgrid)

        # Get the raw RK4 results
        xV = V[idcs.x]
        AV = V[idcs.A]
        BmV = V[idcs.Bm]
        BpV = V[idcs.Bp]
        FV = V[idcs.F]
        rV = V[idcs.r]
        EV = V[idcs.E]

        # Extract the discrete-time update matrices for this time interval
	A_k = reshape(AV, (nx, nx))
        Bm_k = A_k*reshape(BmV, (nx, nu))
        Bp_k = A_k*reshape(BpV, (nx, nu))
        F_k = A_k*reshape(FV, (nx, np))
        r_k = A_k*rV
        E_k = A_k*reshape(EV, size(_E))

        # Save the discrete-time update matrices
        @k(ref.A) = A_k
        @k(ref.Bm) = Bm_k
        @k(ref.Bp) = Bp_k
        @k(ref.F) = F_k
        @k(ref.r) = r_k
        @k(ref.E) = E_k

        # Take this opportunity to comput the defect, which will be needed
        # later for the trust region update
        x_next = @kp1(ref.xd)
        @k(ref.defect) = x_next-xV
        if norm(@k(ref.defect)) > pbm.pars.feas_tol
            ref.feas = false
        end

    end

    return nothing
end

#= Add dynamics constraints to the problem.

Args:
    spbm: the subproblem definition. =#
function _scvx__add_dynamics!(
    spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    N = spbm.scvx.pars.N
    x = spbm.x
    u = spbm.u
    p = spbm.p
    vd = spbm.vd

    for k = 1:N-1
        # Update matrices for this interval
        A =  @k(spbm.ref.A)
        Bm = @k(spbm.ref.Bm)
        Bp = @k(spbm.ref.Bp)
        F =  @k(spbm.ref.F)
        r =  @k(spbm.ref.r)
        E =  @k(spbm.ref.E)

        # Associate matrices with subproblem
        @k(spbm.A) = A
        @k(spbm.Bm) = Bm
        @k(spbm.Bp) = Bp
        @k(spbm.F) = F
        @k(spbm.r) = r
        @k(spbm.E) = E
    end

    # Add dynamics constraint to optimization model
    for k = 1:N-1
        @k(spbm.dynamics) = @constraint(
            spbm.mdl,
            @kp1(x) .== @k(spbm.A)*@k(x)+@k(spbm.Bm)*@k(u)+
            @k(spbm.Bp)*@kp1(u)+@k(spbm.F)*p+@k(spbm.r)+@k(spbm.E)*@k(vd))
    end

    return nothing
end

#= Add boundary condition constraints to the problem.

Args:
    spbm: the subproblem definition. =#
function _scvx__add_bcs!(spbm::SCvxSubproblem)::Nothing
    # Variables and parameters
    traj_pbm = spbm.scvx.traj
    x0 = @first(spbm.x)
    xb0 = @first(spbm.ref.xd)
    xf = @last(spbm.x)
    xbf = @last(spbm.ref.xd)
    p = spbm.p
    pb = spbm.ref.p

    # Initial condition
    gic = traj_pbm.gic(xb0, pb)
    H0 = traj_pbm.H0(xb0, pb)
    K0 = traj_pbm.K0(xb0, pb)
    l0 = gic-H0*xb0-K0*pb
    spbm.vic = @variable(spbm.mdl, [1:length(gic)], base_name="vic")
    spbm.ic = @constraint(spbm.mdl, H0*x0+K0*p+l0+spbm.vic .== 0.0)

    # Terminal condition
    gtc = traj_pbm.gtc(xbf, pb)
    Hf = traj_pbm.Hf(xbf, pb)
    Kf = traj_pbm.Kf(xbf, pb)
    lf = gtc-Hf*xbf-Kf*pb
    spbm.vtc = @variable(spbm.mdl, [1:length(gtc)], base_name="vtc")
    spbm.tc = @constraint(spbm.mdl, Hf*xf+Kf*p+lf+spbm.vtc .== 0.0)

    return nothing
end

#= Add convex state, input, and parameter constraints.

Args:
    spbm: the subproblem definition. =#
function _scvx__add_convex_constraints!(spbm::SCvxSubproblem)::Nothing
    # Variables and parameters
    N = spbm.scvx.pars.N
    traj_pbm = spbm.scvx.traj
    nu = traj_pbm.nu
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Convex state constraints
    for k = 1:N
        X = traj_pbm.X!(@k(x), spbm.mdl)

        if k==1
            # Initialize associated variables
            n_X = length(X)
            spbm.pc_X = T_ConstraintMatrix(undef, n_X, N)
        end

        @k(spbm.pc_X) = X
    end

    # Convex input constraints
    for k = 1:N
        U = traj_pbm.U!(@k(u), spbm.mdl)

        if k==1
            # Initialize associated variables
            n_U = length(U)
            spbm.pc_U = T_ConstraintMatrix(undef, n_U, N)
        end

        @k(spbm.pc_U) = U
    end

    return nothing
end

#= Add non-convex state, input, and parameter constraints.

Args:
    spbm: the subproblem definition. =#
function _scvx__add_nonconvex_constraints!(spbm::SCvxSubproblem)::Nothing
    # Variables and parameters
    N = spbm.scvx.pars.N
    traj_pbm = spbm.scvx.traj
    nu = traj_pbm.nu
    xb = spbm.ref.xd
    ub = spbm.ref.ud
    pb = spbm.ref.p
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Problem-specific convex constraints
    for k = 1:N
        s = traj_pbm.s(@k(xb), @k(ub), pb)
        C = traj_pbm.C(@k(xb), @k(ub), pb)
        D = traj_pbm.D(@k(xb), @k(ub), pb)
        G = traj_pbm.G(@k(xb), @k(ub), pb)
        r = s-C*@k(xb)-D*@k(ub)-G*pb
        lhs = C*@k(x)+D*@k(u)+G*p+r

        if k==1
            # Initialize associated variables
            n_ncvx = length(lhs)
            spbm.vs = @variable(spbm.mdl, [1:n_ncvx, 1:N], base_name="vs")
            spbm.pc_s = T_ConstraintMatrix(undef, n_ncvx, N)
        end

        @k(spbm.pc_s) = @constraint(spbm.mdl, lhs+@k(spbm.vs) .<= 0.0)
    end

    return nothing
end

#= Add trust region constraint to the subproblem.

Args:
    spbm: the subproblem definition. =#
function _scvx__add_trust_region!(spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    N = spbm.scvx.pars.N
    q = spbm.scvx.pars.q_tr
    scale = spbm.scvx.common.scale
    traj_pbm = spbm.scvx.traj
    nx = traj_pbm.nx
    nu = traj_pbm.nu
    η = spbm.η
    sqrt_η = sqrt(η)
    soc_dim = 1+nx+nu
    xh = spbm.xh
    uh = spbm.uh
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    uh_ref = scale.iSu*(spbm.ref.ud.-scale.cu)
    tr_xu = spbm.tr_xu

    # Measure the *scaled* state and input deviations
    dx = xh-xh_ref
    du = uh-uh_ref

    # Trust region constraint
    for k = 1:N
        if q==1
            # 1-norm
            @k(tr_xu) = @constraint(
                spbm.mdl, vcat(η, @k(dx), @k(du))
                in MOI.NormOneCone(soc_dim))
        elseif q==2
            # 2-norm
            cstrt = @constraint(
                spbm.mdl, vcat(spbm.tr_rx[1, k], @k(dx))
                in MOI.SecondOrderCone(1+nx))
            push!(spbm.fit, cstrt)

            cstrt = @constraint(
                spbm.mdl, vcat(spbm.tr_rx[2, k], @k(du))
                in MOI.SecondOrderCone(1+nu))
            push!(spbm.fit, cstrt)

            @k(tr_xu) = @constraint(spbm.mdl, sum(spbm.tr_rx[:, k]) <= η)
        elseif q==4
            # 2-norm squared
            @k(tr_xu) = @constraint(
                spbm.mdl, vcat(sqrt_η, @k(dx), @k(du))
                in MOI.SecondOrderCone(soc_dim))
        else
            # Infinity-norm
            @k(tr_xu) = @constraint(
                spbm.mdl, vcat(η, @k(dx), @k(du))
                in MOI.NormInfinityCone(soc_dim))
        end
    end

    return nothing
end

#= Define the subproblem cost function.

Args:
    spbm: the subproblem definition. =#
function _scvx__add_cost!(spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    N = spbm.scvx.pars.N
    Δτ = spbm.scvx.common.Δτ
    τ_grid = spbm.scvx.common.τ_grid
    x = spbm.x
    u = spbm.u
    p = spbm.p
    P = spbm.P
    Pf = spbm.Pf
    E = spbm.E
    vd = spbm.vd
    vs = spbm.vs
    vic = spbm.vic
    vtc = spbm.vtc

    if isempty(vic) || isempty(vtc)
        err = SCvxError(0, SCVX_EMPTY_VARIABLE,
                        string("Uninitialized boundary condition",
                               " virtual control"))
        throw(err)
    end

    # Original cost
    spbm.L_orig = _scvx__original_cost(x, u, p, spbm.scvx)

    # Virtual control penalty
    spbm.L_pen = trapz(P, τ_grid)+sum(Pf)

    for k = 1:N
        if k<N
            tmp = vcat(@k(P), @k(E)*@k(vd), @k(vs))
        else
            tmp = vcat(@k(P), @k(vs))
        end
        cstrt = @constraint(spbm.mdl, tmp in MOI.NormOneCone(length(tmp)))
        push!(spbm.fit, cstrt)
    end

    pen_ic = @constraint(
        spbm.mdl, vcat(@first(Pf), vic) in MOI.NormOneCone(1+length(vic)))
    push!(spbm.fit, pen_ic)

    pen_tc = @constraint(
        spbm.mdl, vcat(@last(Pf), vtc) in MOI.NormOneCone(1+length(vtc)))
    push!(spbm.fit, pen_tc)

    # Overall cost
    spbm.L = _scvx__overall_cost(spbm.L_orig, spbm.L_pen, spbm.scvx)

    # Associate cost function with the model
    set_objective_function(spbm.mdl, spbm.L)
    set_objective_sense(spbm.mdl, MOI.MIN_SENSE)

    return nothing
end

#= Solve the convex subproblem via numerical optimization.

Args:
    spbm: the subproblem structure. =#
function _scvx__solve_subproblem!(spbm::SCvxSubproblem)::Nothing
    # Optimize
    optimize!(spbm.mdl)

    # Save solution
    spbm.sol = SCvxSubproblemSolution(spbm)

    return nothing
end

#= Check if the subproblem optimization had issues.

A solution is judged unsafe if the numerical optimizer exit code indicates that
there were serious problems in solving the subproblem.

Args:
    sol: the subproblem or directly its solution.

Returns:
    unsafe: true if the subproblem solution process "failed". =#
function _scvx__unsafe_solution(sol::Union{SCvxSubproblemSolution,
                                           SCvxSubproblem})::T_Bool

    # If the parent subproblem passed in, then get its solution
    if typeof(sol)==SCvxSubproblem
        sol = sol.sol
    end

    if !sol.unsafe
        safe = sol.status==MOI.OPTIMAL || sol.status==MOI.ALMOST_OPTIMAL
        sol.unsafe = !safe
    end
    return sol.unsafe
end

#= Check if stopping criterion is triggered.

Args:
    spbm: the subproblem definition.

Returns:
    stop: true if stopping criterion holds. =#
function _scvx__check_stopping_criterion!(spbm::SCvxSubproblem)::T_Bool

    # Extract values
    pbm = spbm.scvx
    N = pbm.pars.N
    ε_abs = pbm.pars.ε_abs
    ε_rel = pbm.pars.ε_rel
    q = pbm.pars.q_exit
    scale = pbm.common.scale
    ref = spbm.ref
    sol = spbm.sol
    xh = scale.iSx*(sol.xd.-scale.cx)
    ph = scale.iSp*(sol.p-scale.cp)
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)

    # Check solution deviation from reference
    dp = norm(ph-ph_ref, q)
    dx = 0.0
    for k = 1:N
        dx = max(dx, norm(@k(xh)-@k(xh_ref), q))
    end
    sol.deviation = dp+dx

    # Check predicted cost improvement
    J_ref = _scvx__solution_cost!(ref, :nonlinear, pbm)
    L_sol = _scvx__solution_cost!(sol, :linear, pbm)
    predicted_improvement = J_ref-L_sol
    sol.dL = predicted_improvement
    dL_relative = sol.dL/abs(J_ref)

    # Compute stopping criterion
    stop = (spbm.iter>1) && ((dL_relative<=ε_rel) || (sol.deviation<=ε_abs))

    return stop
end

#= Compute the new trust region.

Apply the trust region update rule based on the most recent subproblem
solution. This updates the trust region radius, and selects either the current
or the reference solution to act as the next iteration's reference trajectory.

Args:
    spbm: the subproblem definition.

Returns:
    next_ref: reference trajectory for the next iteration.
    next_η: trust region radius for the next iteration. =#
function _scvx__update_trust_region!(
    spbm::SCvxSubproblem)::Tuple{SCvxSubproblemSolution,
                                 T_Real}

    # Parameters
    pbm = spbm.scvx
    sol = spbm.sol
    ref = spbm.ref

    # Compute the actual cost improvement
    J_ref = _scvx__solution_cost!(ref, :nonlinear, pbm)
    J_sol = _scvx__solution_cost!(sol, :nonlinear, pbm)
    actual_improvement = J_ref-J_sol
    sol.dJ = actual_improvement

    # Convexification performance metric
    predicted_improvement = sol.dL
    sol.ρ = actual_improvement/predicted_improvement

    # Apply update rule
    next_ref, next_η = _scvx__update_rule(spbm)

    return next_ref, next_η
end

#= Print command line info message.

Args:
    spbm: the subproblem that was solved.
    err: an SCvx-specific error message. =#
function _scvx__print_info(spbm::SCvxSubproblem,
                           err::Union{Nothing, SCvxError}=nothing)::Nothing

    # Convenience variables
    sol = spbm.sol
    ref = spbm.ref
    table = spbm.scvx.common.table

    if !isnothing(err)
        @printf "ERROR: %s, exiting\n" err.msg
    elseif _scvx__unsafe_solution(sol)
        @printf "ERROR: unsafe solution (%s), exiting\n" sol.status
    else
        # Complicated values
        scale = spbm.scvx.common.scale
        xh = scale.iSx*(sol.xd.-scale.cx)
        uh = scale.iSu*(sol.ud.-scale.cu)
        ph = scale.iSp*(sol.p-scale.cp)
        xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
        uh_ref = scale.iSu*(spbm.ref.ud.-scale.cu)
        ph_ref = scale.iSp*(spbm.ref.p-scale.cp)
        max_dxh = norm(xh-xh_ref, Inf)
        max_duh = norm(uh-uh_ref, Inf)
        max_dph = norm(ph-ph_ref, Inf)
        E = spbm.scvx.common.E
        status = @sprintf "%s" sol.status
        status = status[1:min(8, length(status))]
        ρ = !isnan(sol.ρ) ? @sprintf("%.2f", sol.ρ) : ""

        # Associate values with columns
        assoc = Dict(:iter => spbm.iter,
                     :status => status,
                     :maxvd => norm(sol.vd, Inf),
                     :maxvs => norm(sol.vs, Inf),
                     :maxvbc => norm([sol.vic; sol.vtc], Inf),
                     :cost => sol.J,
                     :dx => max_dxh,
                     :du => max_duh,
                     :dp => max_dph,
                     :dynfeas => sol.feas ? "T" : "F",
                     :δ => sol.deviation,
                     :ρ => ρ,
                     :dL => sol.dL/abs(ref.J)*100,
                     :dtr => sol.tr_update,
                     :rej => sol.reject ? "x" : "",
                     :tr => spbm.η)

        print(assoc, table)
    end

    return nothing
end

#= Add a subproblem to SCvx history.

Args:
    hist: the history.
    spbm: a subproblem structure. =#
function _scvx__save!(hist::SCvxHistory, spbm::SCvxSubproblem)::Nothing
    push!(hist.subproblems, spbm)
    return nothing
end

#= Compute the scaling matrices given the problem definition.

Args:
    pars: the SCvx algorithm parameters.
    traj: the trajectory problem defintion.

Returns:
    scale: the scaling structure. =#
function _scvx__compute_scaling(
    pars::SCvxParameters,
    traj::TrajectoryProblem)::SCvxScaling

    # Parameters
    nx = traj.nx
    nu = traj.nu
    np = traj.np
    solver = pars.solver
    solver_opts = pars.solver_opts
    zero_intvl_tol = sqrt(eps())

    # Map varaibles to these scaled intervals
    intrvl_x = [0.0; 1.0]
    intrvl_u = [0.0; 1.0]

    # >> Compute bounding boxes for state and input <<

    x_bbox = fill(1.0, nx, 2)
    u_bbox = fill(1.0, nu, 2)
    x_bbox[:, 1] .= 0.0
    u_bbox[:, 1] .= 0.0

    defs = [Dict(:dim => nx, :set => traj.X!, :bbox => x_bbox),
            Dict(:dim => nu, :set => traj.U!, :bbox => u_bbox)]

    for def in defs
        for j = 1:2 # 1:min, 2:max
            for i = 1:def[:dim]
                # Initialize JuMP model
                mdl = Model()
                set_optimizer(mdl, solver.Optimizer)
                for (key,val) in solver_opts
                    set_optimizer_attribute(mdl, key, val)
                end
                # Variables
                var = @variable(mdl, [1:def[:dim]])
                # Constraints
                def[:set](var, mdl)
                # Cost
                set_objective_function(mdl, var[i])
                set_objective_sense(mdl, (j==1) ? MOI.MIN_SENSE :
                                    MOI.MAX_SENSE)
                # Solve
                optimize!(mdl)
                # Record the solution
                status = termination_status(mdl)
                if status != MOI.DUAL_INFEASIBLE
                    if (status==MOI.OPTIMAL || status==MOI.ALMOST_OPTIMAL)
                        def[:bbox][i, j] = objective_value(mdl)
                    else
                        msg = "Solver failed during variable scaling (%s)"
                        err = SCvxError(
                            0, SCVX_SCALING_FAILED,
                            @eval @sprintf($msg, $status))
                    end
                end
            end
        end
    end

    # >> Compute scaling matrices and offset vectors <<
    wdth_x = intrvl_x[2]-intrvl_x[1]
    wdth_u = intrvl_u[2]-intrvl_u[1]

    # State scaling terms
    x_min, x_max = x_bbox[:, 1], x_bbox[:, 2]
    diag_Sx = (x_max-x_min)/wdth_x
    diag_Sx[diag_Sx .< zero_intvl_tol] .= 1.0
    Sx = Diagonal(diag_Sx)
    iSx = inv(Sx)
    cx = x_min-diag_Sx*intrvl_x[1]

    # Input scaling terms
    u_min, u_max = u_bbox[:, 1], u_bbox[:, 2]
    diag_Su = (u_max-u_min)/wdth_u
    diag_Su[diag_Su .< zero_intvl_tol] .= 1.0
    Su = Diagonal(diag_Su)
    iSu = inv(Su)
    cu = u_min-diag_Su*intrvl_u[1]

    # Parameter scaling terms (no scaling applied)
    diag_Sp = ones(np)
    Sp = Diagonal(diag_Sp)
    iSp = inv(Sp)
    cp = zeros(np)

    scale = SCvxScaling(Sx, cx, Su, cu, Sp, cp, iSx, iSu, iSp)

    return scale
end

#= Compute concatenanted time derivative vector for dynamics discretization.

Args:
    τ: the time.
    V: the current concatenated vector.
    k: the discrete time grid interval.
    pbm: the SCvx problem definition.
    idcs: indexing arrays into V.
    ref: the reference trajectory.

Returns:
    dVdt: the time derivative of V. =#
function _scvx__derivs(τ::T_Real,
                       V::T_RealVector,
                       k::T_Int,
                       pbm::SCvxProblem,
                       idcs::SCvxDiscretizationIndices,
                       ref::SCvxSubproblemSolution)::T_RealVector
    # Parameters
    nx = pbm.traj.nx
    N = pbm.pars.N
    τ_span = @k(pbm.common.τ_grid, k, k+1)

    # Get current values
    x = V[idcs.x]
    u = linterp(τ, @k(ref.ud, k, k+1), τ_span)
    p = ref.p
    Phi = reshape(V[idcs.A], (nx, nx))
    σ_m = (τ_span[2]-τ)/(τ_span[2]-τ_span[1])
    σ_p = (τ-τ_span[1])/(τ_span[2]-τ_span[1])

    # Compute the state time derivative and local linearization
    f = pbm.traj.f(τ, x, u, p)
    A = pbm.traj.A(τ, x, u, p)
    B = pbm.traj.B(τ, x, u, p)
    F = pbm.traj.F(τ, x, u, p)
    B_m = σ_m*B
    B_p = σ_p*B
    r = f-A*x-B*u-F*p
    E = pbm.common.E

    # Compute the running derivatives for the discrete-time state update
    # matrices
    iPhi = Phi\I(nx)
    dPhidt = A*Phi
    dBmdt = iPhi*B_m
    dBpdt = iPhi*B_p
    dFdt = iPhi*F
    drdt = iPhi*r
    dEdt = iPhi*E

    dVdt = [f; vec(dPhidt); vec(dBmdt); vec(dBpdt);
            vec(dFdt); drdt; vec(dEdt)]

    return dVdt
end

#= Compute the original problem cost function.

Args:
    x: the discrete-time state trajectory.
    u: the discrete-time input trajectory.
    p: the parameter vector.
    pbm: the SCvx problem definition.

Returns:
    cost: the original cost. =#
function _scvx__original_cost(
    x::T_OptiVarMatrix,
    u::T_OptiVarMatrix,
    p::T_OptiVarVector,
    pbm::SCvxProblem)::T_Objective

    # Parameters
    N = pbm.pars.N
    τ_grid = pbm.common.τ_grid
    traj_pbm = pbm.traj

    # Terminal cost
    xf = @last(x)
    J_term = pbm.traj.φ(xf, p)

    # Integrated running cost
    J_run = Vector{T_Objective}(undef, N)
    for k = 1:N
        @k(J_run) = pbm.traj.Γ(@k(x), @k(u), p)
    end
    integ_J_run = trapz(J_run, τ_grid)

    cost = J_term+integ_J_run

    return cost
end

#= Combine original cost and penalty term into an overall cost.

Args:
    orig: the original cost.
    pen: the cost penalty term.
    pbm: the SCvx problem definition

Returns:
    cost: the combined cost. =#
function _scvx__overall_cost(
    orig::T_Objective,
    pen::T_Objective,
    pbm::SCvxProblem)::T_Objective

    # Parameters
    λ = pbm.pars.λ

    # Combined cost
    cost = orig+λ*pen

    return cost
end

#= Compute cost penalty at a particular instant.

This is the integrand of the overall cost penalty term for dynamics and
nonconvex constraint violation.

Note: **this function must match the penalty implemented in
_scvx__add_cost!()**.

Args:
    vd: inconsistency in the dynamics ("defect").
    vs: inconsistency in the nonconvex inequality constraints.

Returns:
    P: the penalty value. =#
function _scvx__P(vd::T_RealVector, vs::T_RealVector)::T_Real
    P = norm(vd, 1)+norm(vs, 1)
    return P
end

#= Compute cost penalty for boundary condition.

This is the cost penalty term for violating a boundary condition.

Note: **this function must match the penalty implemented in
_scvx__add_cost!()**.

Args:
    g: boundary condition value (zero if satisfied).

Returns:
    Pf: the penalty value. =#
function _scvx__Pf(g::T_RealVector)::T_Real
    Pf = norm(g, 1)
    return Pf
end

#= Compute the subproblem cost penalty based on actual constraint violation.

This computes the same cost penalty form as in the subproblem. However, instead
of using virtual control terms, it uses defects from nonlinear propagation of
the dynamics and the actual values of the nonconvex inequality constraints.

If the subproblem solution has already had this function called for it,
re-computation is skipped and the already computed value is returned.

Args:
    sol: the subproblem solution.
    pbm: the SCvx problem definition.
    safe: (optional) whether to check that the coded penalty function
        matches the optimization.

Returns:
    pen: the nonlinear cost penalty term. =#
function _scvx__actual_cost_penalty!(
    sol::SCvxSubproblemSolution,
    pbm::SCvxProblem)::T_Real

    # Values and parameters from the solution
    N = pbm.pars.N
    nx = pbm.traj.nx
    τ_grid = pbm.common.τ_grid
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
        sk = pbm.traj.s(@k(x), @k(u), sol.p)
        @k(P) = _scvx__P(δk, max.(sk, 0.0))
    end
    pen = trapz(P, τ_grid)+_scvx__Pf(gic)+_scvx__Pf(gtc)

    return pen
end

#= Compute the linear or nonlinear overall associated with a solution.

Args:
    sol: the subproblem solution structure.
    kind: whether to compute the linear or nonlinear problem cost.
    pbm: the SCvx problem definition.

Returns:
    cost: the optimal cost associated with this solution. =#
function _scvx__solution_cost!(
    sol::SCvxSubproblemSolution,
    kind::Symbol,
    pbm::SCvxProblem)::T_Real

    if isnan(sol.L_orig)
        sol.L_orig = _scvx__original_cost(
            sol.xd, sol.ud, sol.p, pbm)
    end

    if kind==:linear
        cost = sol.L
    else
        if isnan(sol.J)
            orig = sol.L_orig
            pen = _scvx__actual_cost_penalty!(sol, pbm)
            sol.J = _scvx__overall_cost(orig, pen, pbm)
        end
        cost = sol.J
    end

    return cost
end

#= Apply the low-level SCvx trust region update rule.

This computes the new trust region value and reference trajectory, based on the
obtained subproblem solution.

Args:
    spbm: the subproblem definition.

Returns:
    next_ref: reference trajectory for the next iteration.
    next_η: trust region radius for the next iteration. =#
function _scvx__update_rule(spbm::SCvxSubproblem)::Tuple{
                                SCvxSubproblemSolution,
                                T_Real}
    # Extract relevant data
    pars = spbm.scvx.pars
    sol = spbm.sol
    iter = spbm.iter
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
        next_ref = spbm.ref
        sol.tr_update = "S"
        sol.reject = true
    elseif ρ0<=ρ && ρ<ρ1
        # Mediocre prediction
        next_η = max(η_lb, η/β_sh)
        next_ref = spbm.sol
        sol.tr_update = "S"
        sol.reject = false
    elseif ρ1<=ρ && ρ<ρ2
        # Good prediction
        next_η = η
        next_ref = spbm.sol
        sol.tr_update = ""
        sol.reject = false
    else
        # Excellent prediction
        next_η = min(η_ub, β_gr*η)
        next_ref = spbm.sol
        sol.tr_update = "G"
        sol.reject = false
    end

    return next_ref, next_η
end

#= Mark a solution as unsafe to use.

Args:
    sol: subproblem solution.
    err: the SCvx error that occurred. =#
function _scvx__mark_unsafe!(sol::SCvxSubproblemSolution,
                             err::SCvxError)::Nothing
    sol.status = err.status
    sol.unsafe = true
    return nothing
end

#= Find closest trajectory that satisfies the convex path constraints.

Closeness is measured in an L1-norm set.

Args:
    x_ref: the discrete-time state trajectory to be projected.
    u_ref: the discrete-time input trajectory to be projected.
    pbm: the SCvx problem definition. =#
function _scvx__correct_convex!(
    x_ref::T_RealMatrix,
    u_ref::T_RealMatrix,
    pbm::SCvxProblem)::Nothing

    # Parameters
    N = pbm.pars.N
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    scale = pbm.common.scale
    cone_dim_x = 1+nx
    cone_dim_u = 1+nu

    # Initialize the problem
    opti = SCvxSubproblem(pbm, 0, 0.0)

    # Add the convex path constraints
    _scvx__add_convex_constraints!(opti)

    # Add epigraph constraints to make a convex cost for JuMP
    xh_ref = scale.iSx*(x_ref.-scale.cx)
    uh_ref = scale.iSu*(u_ref.-scale.cu)
    dx = opti.xh-xh_ref
    du = opti.uh-uh_ref
    epi_x = @variable(opti.mdl, [1:N], base_name="τx")
    epi_u = @variable(opti.mdl, [1:N], base_name="τu")
    for k = 1:N
        @constraint(opti.mdl, vcat(@k(epi_x), @k(dx))
                    in MOI.NormOneCone(cone_dim_x))
        @constraint(opti.mdl, vcat(@k(epi_u), @k(du))
                    in MOI.NormOneCone(cone_dim_u))
    end

    # Define the cost
    cost = sum(epi_x)+sum(epi_u)
    set_objective_function(opti.mdl, cost)
    set_objective_sense(opti.mdl, MOI.MIN_SENSE)

    # Solve
    optimize!(opti.mdl)

    # Save solution
    status = termination_status(opti.mdl)
    if (status==MOI.OPTIMAL || status==MOI.ALMOST_OPTIMAL)
        x_ref .= value.(opti.x)
        u_ref .= value.(opti.u)
    else
        msg = string("Solver failed to find the closest initial guess ",
                     "that satisfies the convex constraints (%s)")
        err = SCvxError(0, SCVX_GUESS_PROJECTION_FAILED,
                        @eval @sprintf($msg, $status))
        throw(err)
    end

    return nothing
end
