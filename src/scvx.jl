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
    eps::T_Real       # Convergence tolerance
    feas_tol::T_Real  # Dynamic feasibility tolerance
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
    S::T_IntRange  # Indices for S matrix
    r::T_IntRange  # Indices for r vector
    length::T_Int  # Propagation vector total length
end

#= Subproblem solution.

Structure which stores a solution obtained by solving the subproblem during an
SCvx iteration. =#
mutable struct SCvxSubproblemSolution
    # >> Discrete-time rajectory <<
    xd::T_RealMatrix     # States
    ud::T_RealMatrix     # Inputs
    p::T_RealVector      # Parameter vector
    # >> Virtual control terms <<
    vc::T_RealMatrix     # Virtual control for dynamics
    vb::T_RealMatrix     # Virtual control for path constraints
    P::T_RealVector      # Cost function penalty integrand terms
    # >> Cost function value <<
    J_orig::T_Real       # The original convex cost function
    J_pen::T_Real        # The virtual control penalty
    J_tot::T_Real        # Overall cost function of the subproblem
    J_pen_nl::T_Real     # Nonlinear penalty
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
    S::T_RealTensor      # ... +S[:, :, k]*p+ ...
    r::T_RealMatrix      # ... +r[:, k]
end

#= Common constant terms used throughout the algorithm. =#
struct SCvxCommonConstants
    Δτ::T_Real           # Discrete time step
    τ_grid::T_RealVector # Grid of scaled timed on the [0,1] interval
    scale::SCvxScaling   # Variable scaling
    table::Table         # Iteration info table (printout to REPL)
end

#=Structure which contains all the necessary information to run SCvx.=#
mutable struct SCvxProblem
    pars::SCvxParameters            # Algorithm parameters
    traj::AbstractTrajectoryProblem # The underlying trajectory problem
    consts::SCvxCommonConstants     # Common constant terms
end

#= Subproblem data in JuMP format for the convex numerical optimizer.

This structure holds the final data that goes into the subproblem
optimization. =#
mutable struct SCvxSubproblem
    mdl::Model                   # The optimization problem handle
    # >> Algorithm parameters <<
    scvx::SCvxProblem            # The SCvx problem definition
    # >> Reference and solution trajectories <<
    sol::Union{SCvxSubproblemSolution, Missing} # The solution trajectory
    ref::SCvxSubproblemSolution  # The reference trajectory for linearization
    # >> Cost function <<
    J_orig::T_Objective          # The original convex cost function
    J_pen::T_Objective           # The virtual control penalty
    J_tot::T_Objective           # Overall cost function of the subproblem
    # >> Scaled variables <<
    xh::T_OptiVarMatrix          # Discrete-time states
    uh::T_OptiVarMatrix          # Discrete-time inputs
    ph::T_OptiVarVector          # Parameter
    # >> Physical variables <<
    x::T_OptiVarMatrix           # Discrete-time states
    u::T_OptiVarMatrix           # Discrete-time inputs
    p::T_OptiVarVector           # Parameters
    # >> Virtual control (never scaled) <<
    vc::T_OptiVarMatrix          # Virtual control for dynamics
    vb::T_OptiVarMatrix          # Virtual control for path constraints
    # >> Other variables <<
    P::T_OptiVarVector           # Virtual control penalty function
    # >> Trust region <<
    η::T_Real                    # Trust region radius
    # >> Discrete-time dynamics update matrices <<
    # x[:,k+1] = ...
    A::T_RealTensor              # ...  A[:, :, k]*x[:, k]+ ...
    Bm::T_RealTensor             # ... +Bm[:, :, k]*u[:, k]+ ...
    Bp::T_RealTensor             # ... +Bp[:, :, k]*u[:, k+1]+ ...
    S::T_RealTensor              # ... +S[:, :, k]*p+ ...
    r::T_RealMatrix              # ... +r[:, k]
    # >> Constraint references <<
    dynamics::T_ConstraintMatrix # Dynamics
    ic_x::T_ConstraintVector     # State initial set
    tc_x::T_ConstraintVector     # State target set
    p_bnds::T_ConstraintVector   # Parameter bounds
    pc_x::T_ConstraintMatrix     # Path box constraints on state
    pc_u::T_ConstraintMatrix     # Path box constraints on input
    pc_cvx::T_ConstraintMatrix   # Problem-specific convex path constraints
    pc_ncvx::T_ConstraintMatrix  # Problem-specific nonconvex path constraints
    tr_xu::T_ConstraintVector    # Trust region constraint
    fit::T_ConstraintVector      # Constraints to fit problem and JuMP template
end

# Constant strings for solution exit status
const SCVX_SOLVED = "solved"
const SCVX_FAILED = "failed"

#= Overall trajectory solution.

Structure which holds the trajectory solution that the SCvx algorithm
returns. =#
struct SCvxSolution
    # >> Properties <<
    status::T_String  # Solution status (success? failure?)
    iterations::T_Int # Number of SCvx iterations that occurred
    # >> Discrete-time trajectory <<
    xd::T_RealMatrix  # States
    ud::T_RealMatrix  # Inputs
    p::T_RealVector   # Parameter vector
    # >> Cost function value <<
    J_orig::T_Real    # The original convex cost function
    J_pen::T_Real     # The virtual control penalty
    J_tot::T_Real     # Overall cost function of the subproblem
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
    nx = pbm.traj.vehicle.generic.nx
    nu = pbm.traj.vehicle.generic.nu
    np = pbm.traj.vehicle.generic.np
    id_x  = (1:nx)
    id_A  = id_x[end].+(1:nx*nx)
    id_Bm = id_A[end].+(1:nx*nu)
    id_Bp = id_Bm[end].+(1:nx*nu)
    id_S  = id_Bp[end].+(1:nx*np)
    id_r  = id_S[end].+(1:nx)
    id_sz = length([id_x; id_A; id_Bm; id_Bp; id_S; id_r])
    idcs = SCvxDiscretizationIndices(id_x, id_A, id_Bm, id_Bp, id_S,
                                     id_r, id_sz)
    return idcs
end

#= Construct a subproblem solution from a discrete-time trajectory.

This leaves parameters of the solution other than the passed discrete-time
trajectory unset.

Args:
    x: discrete-time state trajectory.
    u: discrete-time input trajectory.
    p: parameter vector.
    pbm: the SCvx problem definition.

Returns:
    subsol: subproblem solution structure. =#
function SCvxSubproblemSolution(
    x::T_RealMatrix,
    u::T_RealMatrix,
    p::T_RealVector,
    pbm::SCvxProblem)::SCvxSubproblemSolution

    # Parameters
    N = pbm.pars.N
    veh_props = pbm.traj.vehicle.generic
    nx = veh_props.nx
    nu = veh_props.nu
    np = veh_props.np
    n_ncvx = veh_props.n_ncvx

    # Uninitialized parts
    status = MOI.OPTIMIZE_NOT_CALLED
    feas = false
    defect = fill(NaN, nx, N-1)
    deviation = NaN
    unsafe = false
    ρ = NaN
    tr_update = ""
    reject = false

    vc = zeros(nx, N-1)
    vb = zeros(n_ncvx, N)
    P = zeros(N)

    J_orig = NaN
    J_pen = NaN
    J_tot = NaN
    J_pen_nl = NaN

    A = T_RealTensor(undef, nx, nx, N-1)
    Bm = T_RealTensor(undef, nx, nu, N-1)
    Bp = T_RealTensor(undef, nx, nu, N-1)
    S = T_RealTensor(undef, nx, np, N-1)
    r = T_RealMatrix(undef, nx, N-1)

    subsol = SCvxSubproblemSolution(x, u, p, vc, vb, P, J_orig, J_pen, J_tot,
                                    J_pen_nl, status, feas, defect, deviation,
                                    unsafe, ρ, tr_update, reject, A, Bm, Bp, S,
                                    r)

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
                     traj::T)::SCvxProblem where {T<:AbstractTrajectoryProblem}
    # Compute the common constant terms
    scale = _scvx__compute_scaling(traj.bbox)
    τ_grid = LinRange(0.0, 1.0, pars.N)
    Δτ = τ_grid[2]-τ_grid[1]

    # Define the iteration info table
    table = Table([
        # Iteration count
        (:iter, "k", "%d", 2),
        # Solver status
        (:status, "status", "%s", 8),
        # Maximum dynamics virtual control element
        (:maxvc, "vc", "%.2e", 9),
        # Maximum constraints virtual control element
        (:maxvb, "vb", "%.1e", 8),
        # Original cost value
        (:cost, "J", "%.2e", 9),
        # Maximum deviation in state
        (:dx, "Δx", "%.1e", 8),
        # Maximum deviation in input
        (:du, "Δu", "%.1e", 8),
        # Maximum deviation in input
        (:dp, "Δp", "%.1e", 8),
        # Stopping criterion deviation measurement
        (:δ, "δ", "%.1e", 8),
        # Dynamic feasibility flag (true or false)
        (:dynfeas, "dyn", "%s", 3),
        # Trust region size
        (:tr, "η", "%.2f", 6),
        # Convexification performance metrix
        (:ρ, "ρ", "%.2f", 6),
        # Update direction for trust region radius (grow? shrink?)
        (:dtr, "Δη", "%s", 3),
        # Reject solution indicator
        (:rej, "rej", "%s", 5)])

    consts = SCvxCommonConstants(Δτ, τ_grid, scale, table)

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
    sol = SCvxSubproblemSolution(x, u, p, spbm.scvx)

    # Save the optimal cost values
    sol.J_orig = value(spbm.J_orig)
    sol.J_pen = value(spbm.J_pen)
    sol.J_tot = value(spbm.J_tot)

    # Save the virtual control values and penalty term
    sol.vc = value.(spbm.vc)
    sol.vb = value.(spbm.vb)
    sol.P = value.(spbm.P)

    # Save the solution status
    sol.status = termination_status(spbm.mdl)

    return sol
end

#= Constructor for an empty convex optimization subproblem.

No cost or constraints. Just the decision variables and empty associated
parameters.

Args:
    pbm: the SCvx problem being solved.
    ref: the reference trajectory.
    η: the trust region radius.

Returns:
    spbm: the subproblem structure. =#
function SCvxSubproblem(pbm::SCvxProblem,
                        ref::SCvxSubproblemSolution,
                        η::T_Real)::SCvxSubproblem
    # Sizes
    nx = pbm.traj.vehicle.generic.nx
    nu = pbm.traj.vehicle.generic.nu
    np = pbm.traj.vehicle.generic.np
    n_cvx = pbm.traj.vehicle.generic.n_cvx
    n_ncvx = pbm.traj.vehicle.generic.n_ncvx
    N = pbm.pars.N

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
    J_orig = missing
    J_pen = missing
    J_tot = missing

    # Decision variables (scaled)
    xh = @variable(mdl, [1:nx, 1:N], base_name="xh")
    uh = @variable(mdl, [1:nu, 1:N], base_name="uh")
    ph = @variable(mdl, [1:np], base_name="ph")

    # Physical decision variables
    x = pbm.consts.scale.Sx*xh.+pbm.consts.scale.cx
    u = pbm.consts.scale.Su*uh.+pbm.consts.scale.cu
    p = pbm.consts.scale.Sp*ph.+pbm.consts.scale.cp
    vc = @variable(mdl, [1:nx, 1:N-1], base_name="vc")
    vb = @variable(mdl, [1:n_ncvx, 1:N], base_name="vb")

    # Other variables
    P = @variable(mdl, [1:N], base_name="P")

    # Uninitialized parameters
    A = T_RealTensor(undef, nx, nx, N-1)
    Bm = T_RealTensor(undef, nx, nu, N-1)
    Bp = T_RealTensor(undef, nx, nu, N-1)
    S = T_RealTensor(undef, nx, np, N-1)
    r = T_RealMatrix(undef, nx, N-1)

    # Empty constraints
    dynamics = T_ConstraintMatrix(undef, nx, N-1)
    ic_x = T_ConstraintVector(undef, nx)
    tc_x = T_ConstraintVector(undef, nx)
    p_bnds = T_ConstraintVector(undef, np)
    pc_x = T_ConstraintMatrix(undef, nx, N)
    pc_u = T_ConstraintMatrix(undef, nu, N)
    pc_cvx = T_ConstraintMatrix(undef, n_cvx, N)
    pc_ncvx = T_ConstraintMatrix(undef, n_ncvx, N)
    tr_xu = T_ConstraintVector(undef, N)
    fit = T_ConstraintVector(undef, N)

    spbm = SCvxSubproblem(mdl, pbm, sol, ref, J_orig, J_pen, J_tot, xh, uh, ph,
                          x, u, p, vc, vb, P, η, A, Bm, Bp, S, r, dynamics,
                          ic_x, tc_x, p_bnds, pc_x, pc_u, pc_cvx, pc_ncvx,
                          tr_xu, fit)

    return spbm
end

#= Convert subproblem solution to a final trajectory solution.

This is what the SCvx algorithm returns in the end to the user.

Args:
    history: SCvx iteration history.

Returns:
    sol: the trajectory solution. =#
function SCvxSolution(history::SCvxHistory)::SCvxSolution
    last_spbm = history.subproblems[end]
    last_sol = last_spbm.sol
    num_iters = _scvx__scvx_iter_count(history)

    if unsafe_solution(last_sol)
        # SCvx failed :(
        status = @sprintf "%s (%s)" SCVX_FAILED last_sol.status
        xd = T_RealMatrix(undef, size(last_sol.xd))
        ud = T_RealMatrix(undef, size(last_sol.ud))
        p = T_RealVector(undef, size(last_sol.p))
        J_orig = Inf
        J_pen = Inf
        J_tot = Inf
    else
        # SCvx solved the problem!
        status = SCVX_SOLVED
        xd = last_sol.xd
        ud = last_sol.ud
        p = last_sol.p
        J_orig = last_sol.J_orig
        J_pen = last_sol.J_pen
        J_tot = last_sol.J_tot
    end

    sol = SCvxSolution(status, num_iters, xd, ud, p, J_orig, J_pen, J_tot)

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

#= Compute the initial trajectory guess.

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to an SCvxSubproblemSolution structure.

Args:
    pbm: the SCvx problem structure.

Returns:
    guess: the initial guess. =#
function generate_initial_guess(pbm::SCvxProblem)::SCvxSubproblemSolution
    x, u, p = initial_guess(pbm.traj, pbm.pars.N)
    guess = SCvxSubproblemSolution(x, u, p, pbm)
    return guess
end

#= Discrete linear time varying dynamics computation.

Compute the discrete-time update matrices for the linearized dynamics about a
reference trajectory. As a byproduct, this calculates the defects needed for
the trust region update.

Args:
    ref: the reference trajectory for which the propagation is done.
    pbm: the SCvx problem definition. =#
function discretize!(ref::SCvxSubproblemSolution,
                     pbm::SCvxProblem)::Nothing
    # Parameters
    nx = pbm.traj.vehicle.generic.nx
    nu = pbm.traj.vehicle.generic.nu
    np = pbm.traj.vehicle.generic.np
    N = pbm.pars.N
    Nsub = pbm.pars.Nsub

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
        f(τ::T_Real, V::T_RealVector)::T_RealVector =
            _scvx__derivs(τ, V, k, pbm, idcs, ref)
        τ_subgrid = T_RealVector(
            LinRange(pbm.consts.τ_grid[k], pbm.consts.τ_grid[k+1], Nsub))
        V = rk4(f, V0, τ_subgrid)

        # Get the raw RK4 results
        xV = V[idcs.x]
        AV = V[idcs.A]
        BmV = V[idcs.Bm]
        BpV = V[idcs.Bp]
        SV = V[idcs.S]
        rV = V[idcs.r]

        # Extract the discrete-time update matrices for this time interval
	A_k = reshape(AV, (nx, nx))
        Bm_k = A_k*reshape(BmV, (nx, nu))
        Bp_k = A_k*reshape(BpV, (nx, nu))
        S_k = A_k*reshape(SV, (nx, np))
        r_k = A_k*rV

        # Save the discrete-time update matrices
        @k(ref.A) = A_k
        @k(ref.Bm) = Bm_k
        @k(ref.Bp) = Bp_k
        @k(ref.S) = S_k
        @k(ref.r) = r_k

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
function add_dynamics!(
    spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    N = spbm.scvx.pars.N
    x = spbm.x
    u = spbm.u
    p = spbm.p
    vc = spbm.vc

    for k = 1:N-1
        # Update matrices for this interval
        A =  @k(spbm.ref.A)
        Bm = @k(spbm.ref.Bm)
        Bp = @k(spbm.ref.Bp)
        S =  @k(spbm.ref.S)
        r =  @k(spbm.ref.r)

        # Associate matrices with subproblem
        @k(spbm.A) = A
        @k(spbm.Bm) = Bm
        @k(spbm.Bp) = Bp
        @k(spbm.S) = S
        @k(spbm.r) = r
    end

    # Add dynamics constraint to optimization model
    for k = 1:N-1
        @k(spbm.dynamics) = @constraint(
            spbm.mdl,
            @kp1(x) .== @k(spbm.A)*@k(x)+@k(spbm.Bm)*@k(u)+
            @k(spbm.Bp)*@kp1(u)+@k(spbm.S)*p+@k(spbm.r)+@k(vc))
    end

    return nothing
end

#= Add boundary condition constraints to the problem.

Args:
    spbm: the subproblem definition. =#
function add_bcs!(spbm::SCvxSubproblem)::Nothing
    # Variables and parameters
    x0 = @first(spbm.x)
    xf = @last(spbm.x)
    p = spbm.p
    bbox = spbm.scvx.traj.bbox

    # Initial condition
    spbm.ic_x = @constraint(
        spbm.mdl,
        bbox.init.x.min .<= x0 .<= bbox.init.x.max)

    # Terminal condition
    spbm.tc_x = @constraint(
        spbm.mdl,
        bbox.trgt.x.min .<= xf .<= bbox.trgt.x.max)

    # Parameter value constraints
    spbm.p_bnds = @constraint(
        spbm.mdl,
        bbox.path.p.min .<= p .<= bbox.path.p.max)

    return nothing
end

#= Add convex state, input, and parameter constraints.

Args:
    spbm: the subproblem definition. =#
function add_convex_constraints!(spbm::SCvxSubproblem)::Nothing
    # Variables and parameters
    N = spbm.scvx.pars.N
    traj_pbm = spbm.scvx.traj
    nu = traj_pbm.vehicle.generic.nu
    bbox = traj_pbm.bbox
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Path box constrains on the state
    for k = 1:N
        @k(spbm.pc_x) = @constraint(
            spbm.mdl,
            bbox.path.x.min .<= @k(x) .<= bbox.path.x.max)
    end

    # Path box constrains on the input
    for k = 1:N
        @k(spbm.pc_u) = @constraint(
            spbm.mdl,
            bbox.path.u.min .<= @k(u) .<= bbox.path.u.max)
    end

    # Problem-specific convex constraints
    for k = 1:N
        constraints = add_mdl_cvx_constraints!(
            @k(x), @k(u), p, spbm.mdl, traj_pbm)
        @k(spbm.pc_cvx) = constraints
    end

    return nothing
end

#= Add non-convex state, input, and parameter constraints.

Args:
    spbm: the subproblem definition. =#
function add_nonconvex_constraints!(spbm::SCvxSubproblem)::Nothing
    # Variables and parameters
    N = spbm.scvx.pars.N
    traj_pbm = spbm.scvx.traj
    nu = traj_pbm.vehicle.generic.nu
    bbox = traj_pbm.bbox
    x_ref = spbm.ref.xd
    u_ref = spbm.ref.ud
    p_ref = spbm.ref.p
    x = spbm.x
    u = spbm.u
    p = spbm.p
    vb = spbm.vb

    # Problem-specific convex constraints
    for k = 1:N
        constraints = add_mdl_ncvx_constraint!(
            @k(x), @k(u), p, @k(x_ref), @k(u_ref), p_ref, @k(vb),
            spbm.mdl, traj_pbm)
        # @k(spbm.pc_ncvx) = constraints
    end
end

#= Add trust region constraint to the subproblem.

Args:
    spbm: the subproblem definition. =#
function add_trust_region!(spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    N = spbm.scvx.pars.N
    scale = spbm.scvx.consts.scale
    vehicle = spbm.scvx.traj.vehicle
    nx = vehicle.generic.nx
    nu = vehicle.generic.nu
    sqrt_η = sqrt(spbm.η)
    soc_dim = 1+nx+nu
    xh = spbm.xh
    uh = spbm.uh
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    uh_ref = scale.iSu*(spbm.ref.ud.-scale.cu)
    tr_xu = spbm.tr_xu

    # Trust region constraint
    # Note that we measure the *scaled* state and input deviations
    dx = xh-xh_ref
    du = uh-uh_ref
    for k = 1:N
        @k(tr_xu) = @constraint(
            spbm.mdl, vcat(sqrt_η, @k(dx), @k(du))
            in MOI.SecondOrderCone(soc_dim))
        # @k(tr_xu) = @constraint(
        #     spbm.mdl, vcat(spbm.η, @k(dx), @k(du))
        #     in MOI.NormInfinityCone(soc_dim))
    end

    return nothing
end

#= Define the subproblem cost function.

Args:
    spbm: the subproblem definition. =#
function add_cost!(spbm::SCvxSubproblem)::Nothing

    # Variables and parameters
    N = spbm.scvx.pars.N
    Δτ = spbm.scvx.consts.Δτ
    τ_grid = spbm.scvx.consts.τ_grid
    traj_pbm = spbm.scvx.traj
    x = spbm.x
    u = spbm.u
    p = spbm.p
    P = spbm.P
    vc = spbm.vc
    vb = spbm.vb
    vc_sz = length(vc)
    vb_sz = length(vb)

    # >> The cost function <<

    # Original cost
    spbm.J_orig = _scvx__original_cost(x, u, p, spbm.scvx)

    # Virtual control penalty
    for k = 1:N
        if k < N
            tmp = vcat(@k(P), vec(@k(vc)), vec(@k(vb)))
        else
            tmp = vcat(@k(P), vec(@k(vb)))
        end
        @k(spbm.fit) = @constraint(
            spbm.mdl, tmp in MOI.NormOneCone(length(tmp)))
    end
    spbm.J_pen = trapz(P, τ_grid)

    # Overall cost
    spbm.J_tot = _scvx__overall_cost(spbm.J_orig, spbm.J_pen, spbm.scvx)

    # Associate cost function with the model
    set_objective_function(spbm.mdl, spbm.J_tot)
    set_objective_sense(spbm.mdl, MOI.MIN_SENSE)

    return nothing
end

#= Solve the convex subproblem via numerical optimization.

Args:
    spbm: the subproblem structure. =#
function solve_subproblem!(spbm::SCvxSubproblem)::Nothing
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
function unsafe_solution(sol::Union{SCvxSubproblemSolution,
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
function check_stopping_criterion!(spbm::SCvxSubproblem)::T_Bool

    # Extract values
    N = spbm.scvx.pars.N
    eps = spbm.scvx.pars.eps
    scale = spbm.scvx.consts.scale
    sol = spbm.sol
    xh = scale.iSx*(sol.xd.-scale.cx)
    ph = scale.iSp*(sol.p-scale.cp)
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)

    # Check dynamic feasibility (discretizes the dynamics as a byproduct)
    discretize!(sol, spbm.scvx)
    dyn_feas = sol.feas

    # Check solution deviation from reference
    q = Inf
    dp = norm(ph-ph_ref, q)
    dx = 0.0
    for k = 1:N
        dx = max(dx, norm(@k(xh)-@k(xh_ref), q))
    end
    sol.deviation = dp+dx

    # Compute stopping criterion
    stop = dyn_feas && (sol.deviation<=eps)

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
function update_trust_region!(spbm::SCvxSubproblem)::Tuple{
    SCvxSubproblemSolution,
    T_Real}

    # Parameters
    pbm = spbm.scvx
    sol = spbm.sol
    ref = spbm.ref

    # Compute the actual cost improvement
    J_ref = _scvx__solution_cost!(ref, :nonlinear, pbm)
    J_sol = _scvx__solution_cost!(sol, :nonlinear, pbm)
    actual_improvement = J_ref-J_sol

    # Compute the predicted cost improvement
    L_sol = _scvx__solution_cost!(sol, :linear, pbm)
    predicted_improvement = J_ref-L_sol

    # Convexification performance metric
    sol.ρ = actual_improvement/predicted_improvement

    # Apply update rule
    next_ref, next_η = _scvx__update_rule(sol.ρ, spbm)

    return next_ref, next_η
end

#= Print command line info message.

Args:
    history: the SCvx iteration history.
    pvm: the SCvx problem definition.
    err: an SCvx-specific error message. =#
function print_info(history::SCvxHistory,
                    pbm::SCvxProblem,
                    err::Union{Nothing, SCvxError}=nothing)::Nothing

    # Convenience variables
    spbm = history.subproblems[end]
    sol = spbm.sol
    table = pbm.consts.table

    if !isnothing(err)
        @printf "ERROR: %s, exiting\n" err.msg
    elseif unsafe_solution(sol)
        @printf "ERROR: unsafe solution (%s), exiting\n" sol.status
    else
        # Complicated values
        scale = spbm.scvx.consts.scale
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

        # Associate values with columns
        assoc = Dict(:iter => _scvx__scvx_iter_count(history),
                     :status => status,
                     :maxvc => norm(sol.vc, Inf),
                     :maxvb => norm(sol.vb, Inf),
                     :cost => sol.J_orig,
                     :dx => max_dxh,
                     :du => max_duh,
                     :dp => max_dph,
                     :dynfeas => sol.feas ? "T" : "F",
                     :δ => sol.deviation,
                     :ρ => sol.ρ,
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
function save!(hist::SCvxHistory, spbm::SCvxSubproblem)::Nothing
    push!(hist.subproblems, spbm)
    return nothing
end

# ..:: Private methods ::..

#= Compute the scaling matrices given the problem definition.

Args:
    bbox: the trajectory bounding box.

Returns:
    scale: the scaling structure. =#
function _scvx__compute_scaling(bbox::TrajectoryBoundingBox)::SCvxScaling
    # Parameters
    nx = length(bbox.path.x.min)
    nu = length(bbox.path.u.min)
    zero_intvl_tol = sqrt(eps())

    # State, control and final time "box" bounds
    x_min = bbox.path.x.min
    x_max = bbox.path.x.max
    u_min = bbox.path.u.min
    u_max = bbox.path.u.max
    p_min = bbox.path.p.min
    p_max = bbox.path.p.max

    # Choose [0, 1] box for scaled variable intervals
    intrvl_x = [0; 1]
    intrvl_u = [0; 1]
    intrvl_p = [0; 1]
    wdth_x   = intrvl_x[2]-intrvl_x[1]
    wdth_u   = intrvl_u[2]-intrvl_u[1]
    wdth_p   = intrvl_p[2]-intrvl_p[1]

    # State scaling terms
    diag_Sx = (x_max-x_min)/wdth_x
    diag_Sx[diag_Sx .< zero_intvl_tol] .= 1.0
    Sx = Diagonal(diag_Sx)
    iSx = inv(Sx)
    cx = x_min-diag_Sx*intrvl_x[1]

    # Input scaling terms
    diag_Su = (u_max-u_min)/wdth_u
    diag_Su[diag_Su .< zero_intvl_tol] .= 1.0
    Su = Diagonal(diag_Su)
    iSu = inv(Su)
    cu = u_min-diag_Su*intrvl_u[1]

    # Temporal (parameter) scaling terms
    diag_Sp = (p_max-p_min)/wdth_p
    diag_Sp[diag_Sp .< zero_intvl_tol] .= 1.0
    Sp = Diagonal(diag_Sp)
    iSp = inv(Sp)
    cp = p_min-diag_Sp*intrvl_p[1]

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
    nx = pbm.traj.vehicle.generic.nx
    N = pbm.pars.N
    τ_span = @k(pbm.consts.τ_grid, k, k+1)

    # Get current values
    x_now = V[idcs.x]
    u_now = linterp(τ, @k(ref.ud, k, k+1), τ_span)
    p = ref.p
    Phi = reshape(V[idcs.A], (nx, nx))
    λ_m = (τ_span[2]-τ)/(τ_span[2]-τ_span[1])
    λ_p = (τ-τ_span[1])/(τ_span[2]-τ_span[1])

    # Compute the state time derivative and local linearization
    f = dynamics(pbm.traj, τ, x_now, u_now, p)
    A, B, S = jacobians(pbm.traj, τ, x_now, u_now, p)
    B_m = λ_m*B
    B_p = λ_p*B
    r = f-A*x_now-B*u_now-S*p

    # Compute the running derivatives for the discrete-time state update
    # matrices
    iPhi = Phi\I(nx)
    dPhidt = A*Phi
    dBmdt = iPhi*B_m
    dBpdt = iPhi*B_p
    dSdt = iPhi*S
    drdt = iPhi*r

    dVdt = [f; vec(dPhidt); vec(dBmdt); vec(dBpdt); vec(dSdt); drdt]

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
    τ_grid = pbm.consts.τ_grid
    traj_pbm = pbm.traj

    # Terminal cost
    xf = @last(x)
    J_term = terminal_cost(xf, p, traj_pbm)

    # Integrated running cost
    J_run = Vector{T_Objective}(undef, N)
    for k = 1:N
        @k(J_run) = running_cost(@k(x), @k(u), p, traj_pbm)
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

This is the integrand of the overall cost penalty term.

Args:
    δ: inconsistency in the dynamics ("defect").
    s: inconsistency in the nonconvex inequality constraints (value of the
        constraint left-hand side).

Returns:
    P: the penalty value. =#
function _scvx__P(δ::T_RealVector, s::T_RealVector)::T_Real
    s_plus = max.(s, 0.0)
    P = norm(δ, 1)+norm(s_plus, 1)
    return P
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

Returns:
    pen: the nonlinear cost penalty term. =#
function _scvx__actual_cost_penalty!(
    sol::SCvxSubproblemSolution,
    pbm::SCvxProblem)::T_Real

    if isnan(sol.J_pen_nl)
        # >> Compute the nonlinear penalty <<

        _scvx__assert_penalty_match!(sol, pbm)

        # Parameters
        N = pbm.pars.N
        nx = pbm.traj.vehicle.generic.nx
        n_ncvx = pbm.traj.vehicle.generic.n_ncvx
        τ_grid = pbm.consts.τ_grid

        # Values from the solution
        δ = sol.defect
        s = T_RealMatrix(undef, n_ncvx, N)
        for k = 1:N
            @k(s), _, _, _ = ncvx_constraint(
                @k(sol.xd), @k(sol.ud), sol.p, pbm.traj)
        end

        # Integrate the nonlinear penalty term
        P = T_RealVector(undef, N)
        for k = 1:N
            δk = (k<N) ? @k(δ) : zeros(nx)
            @k(P) = _scvx__P(δk, @k(s))
        end
        pen = trapz(P, τ_grid)

        sol.J_pen_nl = pen
    else
        # >> Retrieve existing penalty value <<
        pen = sol.J_pen_nl
    end

    return pen
end

#= Compute the linear or nonlinear overall associated with a solution.

Args:
    sol: the subproblem solution structure.

Returns:
    cost: the optimal cost associated with this solution. =#
function _scvx__solution_cost!(
    sol::SCvxSubproblemSolution,
    kind::Symbol,
    pbm::SCvxProblem)::T_Real

    # Original cost
    if isnan(sol.J_orig)
        x = sol.xd
        u = sol.ud
        p = sol.p
        sol.J_orig = _scvx__original_cost(x, u, p, pbm)
    end
    orig = sol.J_orig

    # Cost penalty term
    pen = (kind==:linear) ? sol.J_pen : _scvx__actual_cost_penalty!(sol, pbm)

    # Overall cost
    cost = _scvx__overall_cost(orig, pen, pbm)

    return cost
end

#= Apply the low-level SCvx trust region update rule.

This computes the new trust region value and reference trajectory, based on the
obtained subproblem solution.

Args:
    ρ: the convexification performance metric.
    spbm: the subproblem definition.

Returns:
    next_ref: reference trajectory for the next iteration.
    next_η: trust region radius for the next iteration. =#
function _scvx__update_rule(ρ::T_Real,
                            spbm::SCvxSubproblem)::Tuple{
                                SCvxSubproblemSolution,
                                T_Real}
    # Extract relevant data
    pars = spbm.scvx.pars
    sol = spbm.sol
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
        # OK prediction
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

#= Assert that the penalty function computed by JuMP is correct.

The JuMP formulation does not allow using _scvx__P() directly. We had to use a
tight relaxation instead. This function checks that _scvx__P() matches what
JuMP used, which makes sure that there is no formulation discrepancy.

Args:
    sol: the subproblem solution.
    pbm: the SCvx problem definition.

Returns:
    match: indicator that everything is correct.

Raises:
    SCvxError: if the penalty match fails. =#
function _scvx__assert_penalty_match!(sol::SCvxSubproblemSolution,
                                      pbm::SCvxProblem)::Nothing
    # Parameters
    N = pbm.pars.N
    nx = pbm.traj.vehicle.generic.nx

    # A small number for the equality tolerance
    # We don't want to make this a user parameter, because ideally this
    # assertion is "built-in", abstracted from the user
    check_tol = 1e-5

    # Variables
    vc = sol.vc
    vb = sol.vb
    P_jump = sol.P

    for k = 1:N
        vck = (k<N) ? @k(vc) : zeros(nx)
        Pk_coded = _scvx__P(abs.(vck), abs.(@k(vb)))
        Pk_jump = @k(P_jump)

        if abs(Pk_coded-Pk_jump) > check_tol
            # Create an SCvxError
            fmt = string("The coded penalty value (%.3e) does not match ",
                         "the JuMP penalty value (%.3e) at time step %d")
            msg = @eval @sprintf($fmt, $Pk_coded, $Pk_jump, $k)
            err = SCvxError(k, PENALTY_CHECK_FAILED, msg)

            # Mark the solution as unsafe
            _scvx__mark_unsafe!(sol, err)

            throw(err)
        end
    end

    return nothing
end

#= Count how many SCvx iterations have happened.

Args:
    history: the SCvx iteration history.

Returns:
    num_iter: the number of SCvx iterations. =#
function _scvx__scvx_iter_count(history::SCvxHistory)::T_Int
    num_iter = length(history.subproblems)
    return num_iter
end
