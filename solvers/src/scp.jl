#= Data structures and methods common across SCP algorithms.

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

if isdefined(@__MODULE__, :LanguageServer)
    include("../../utils/src/Utils.jl")
    include("../../parser/src/Parser.jl")
    using .Utils
    using .Utils: linterp
    using .Utils.Types: sample
    using .Parser
    using .Parser.TrajectoryProblem
    import .Parser.ConicLinearProgram: @add_constraint, @new_variable
    import .Parser.ConicLinearProgram: ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
    import .Parser.ConicLinearProgram: QuadraticCost
    import .Parser.ConicLinearProgram: solve!
end

using Utils
using Parser

using LinearAlgebra
using JuMP
using Printf

export SCPSolution, SCPHistory

const ST = Types
const CLP = ConicLinearProgram

const RealTypes = ST.RealTypes
const IntRange = ST.IntRange
const RealVector = ST.RealVector
const RealMatrix = ST.RealMatrix
const Trajectory = ST.ContinuousTimeTrajectory
const Objective = Union{ST.Objective, QuadraticCost}
const VariableVector = ST.VariableVector
const VariableMatrix = ST.VariableMatrix

abstract type SCPParameters end
abstract type SCPSubproblem end
abstract type SCPSubproblemSolution end

""" Variable scaling parameters.

Holds the SCP subproblem internal scaling parameters, which makes the numerical
optimization subproblems better conditioned.
"""
struct SCPScaling
    Sx::RealMatrix  # State scaling coefficient matrix
    cx::RealVector  # State scaling offset vector
    Su::RealMatrix  # Input scaling coefficient matrix
    cu::RealVector  # Input scaling offset vector
    Sp::RealMatrix  # Parameter scaling coefficient matrix
    cp::RealVector  # Parameter scaling offset matrix
    iSx::RealMatrix # Inverse of state scaling matrix
    iSu::RealMatrix # Inverse of input scaling matrix
    iSp::RealMatrix # Inverse of parameter scaling coefficient matrix
end # struct

""" Indexing arrays for convenient access during dynamics discretization.

Container of indices useful for extracting variables from the propagation
vector during the linearized dynamics discretization process.
"""
struct SCPDiscretizationIndices
    x::IntRange  # Indices for state
    A::IntRange  # Indices for A matrix
    Bm::IntRange # Indices for B_{-} matrix
    Bp::IntRange # Indices for B_{+} matrix
    F::IntRange  # Indices for S matrix
    r::IntRange  # Indices for r vector
    E::IntRange  # Indices for E matrix
    length::Int  # Propagation vector total length
end # struct

"""" Common constant terms used throughout the algorithm."""
struct SCPCommon
    # >> Discrete-time grid <<
    Δt::RealTypes        # Discrete time step
    t_grid::RealVector   # Grid of scaled timed on the [0,1] interval
    E::RealMatrix        # Continuous-time matrix for dynamics virtual control
    scale::SCPScaling    # Variable scaling
    id::SCPDiscretizationIndices # Convenience indices during propagation
    table::ST.Table      # Iteration info table (printout to REPL)
end # struct

""" Structure which contains all the necessary information to run SCP."""
struct SCPProblem{T<:SCPParameters}
    pars::T                 # Algorithm parameters
    traj::TrajectoryProblem # The underlying trajectory problem
    common::SCPCommon       # Common precomputed terms

    """"
        SCPProblem(pars, traj, common)

    Basic constructor.

    # Arguments
    - See the above comments.

    # Returns
    - `pbm`: the SCP problem definition structure.
    """
    function SCPProblem(pars::T,
                        traj::TrajectoryProblem,
                        common::SCPCommon)::SCPProblem where {T<:SCPParameters}
        if !(traj.nx>=1 && traj.nu >=1)
            msg = string("the current implementation only supports",
                         " problems with at least 1 state and 1 control")
            err = SCPError(0, SCP_BAD_PROBLEM, msg)
            throw(err)
        end

        traj.scp = pars # Associate SCP parameters with the problem

        pbm = new{typeof(pars)}(pars, traj, common)

        return pbm
    end # function
end # struct

""" Overall trajectory solution.

Structure which holds the trajectory solution that the SCP algorithm returns.
"""
struct SCPSolution
    # >> Properties <<
    status::String    # Solution status (success? failure?)
    algo::String      # Which algorithm was used to obtain this solution
    iterations::Int   # Number of SCP iterations that occurred
    cost::RealTypes   # The original convex cost function
    # >> Discrete-time trajectory <<
    td::RealVector  # Discrete times
    xd::RealMatrix  # States
    ud::RealMatrix  # Inputs
    p::RealVector   # Parameter vector
    # >> Continuous-time trajectory <<
    xc::Union{Trajectory, Missing} # States
    uc::Union{Trajectory, Missing} # Inputs
end # struct

"""" SCP iteration history data."""
struct SCPHistory{T<:SCPSubproblem}
    subproblems::Vector{T} # Subproblems
end # struct

"""
    SCPDiscretizationIndices(traj, E)

Indexing arrays from problem definition.

# Arguments
- `traj`: the trajectory problem definition.
- `E`: the dynamics virtual control coefficient matrix.

# Returns
- `idcs`: the indexing array structure.
"""
function SCPDiscretizationIndices(
    traj::TrajectoryProblem,
    E::RealMatrix)::SCPDiscretizationIndices

    nx = traj.nx
    nu = traj.nu
    np = traj.np
    id_x  = (1:nx)
    id_A  = id_x[end].+(1:nx*nx)
    id_Bm = id_A[end].+(1:nx*nu)
    id_Bp = id_Bm[end].+(1:nx*nu)
    id_S  = id_Bp[end].+(1:nx*np)
    id_r  = id_S[end].+(1:nx)
    id_E  = id_r[end].+(1:length(E))
    id_sz = length([id_x; id_A; id_Bm; id_Bp; id_S; id_r; id_E])
    idcs = SCPDiscretizationIndices(id_x, id_A, id_Bm, id_Bp, id_S,
                                    id_r, id_E, id_sz)

    return idcs
end # function

"""
    SCPProblem(pars, traj, table)

Construct the SCP problem definition. This internally also computes the scaling
matrices used to improve subproblem numerics.

# Arguments
- `pars`: GuSTO algorithm parameters.
- `traj`: the underlying trajectory optimization problem.
- `table`: the iteration progress table.

# Returns
- `pbm`: the problem structure ready for being solved by GuSTO.
"""
function SCPProblem(
    pars::T,
    traj::TrajectoryProblem,
    table::ST.Table)::SCPProblem where {T<:SCPParameters}

    # Compute the common constant terms
    t_grid = RealVector(LinRange(0.0, 1.0, pars.N))
    Δt = t_grid[2]-t_grid[1]
    E = RealMatrix(collect(Int, I(traj.nx)))
    scale = compute_scaling(pars, traj, t_grid)
    idcs = SCPDiscretizationIndices(traj, E)
    consts = SCPCommon(Δt, t_grid, E, scale, idcs, table)

    pbm = SCPProblem(pars, traj, consts)

    return pbm
end # function

"""
    SCPSubproblemSolution!(spbm)

Create the subproblem solution structure. This calls the SCP algorithm-specific
function, and also saves some general properties of the solution.

# Arguments
- `spbm`: the subproblem structure.
"""
function SCPSubproblemSolution!(spbm::T)::Nothing where {T<:SCPSubproblem}
    # Save the solution
    # (this complicated-looking thing calls the constructor for the
    #  SCPSubproblemSolution child type)
    constructor = Meta.parse(string(typeof(spbm.ref)))
    spbm.sol = eval(Expr(:call, constructor, spbm))

    # Save common solution properties
    spbm.sol.status = termination_status(spbm.__prg)

    # Save statistics about the subproblem and its solution
    spbm.timing[:solve] = solve_time(spbm.__prg)
    spbm.timing[:discretize] = spbm.sol.dyn.timing

    return nothing
end # function

"""
    SCPSolution(history)

Convert subproblem solution to a final trajectory solution. This is what the
SCP algorithm returns in the end to the user.

# Arguments
- `history`: SCP iteration history.
- `scp_algo`: name of the SCP algorithm used.

# Returns
- `sol`: the trajectory solution.
"""
function SCPSolution(history::SCPHistory)::SCPSolution

    # Get the solution
    last_spbm = history.subproblems[end]
    last_sol = last_spbm.sol

    # Extract relevant parameters
    num_iters = last_spbm.iter
    pbm = last_spbm.def
    N = pbm.pars.N
    Nsub = pbm.pars.Nsub
    td = pbm.common.t_grid
    algo = last_spbm.algo

    if unsafe_solution(last_sol)
        # SCP failed :(
        status = @sprintf "%s (%s)" SCP_FAILED last_sol.status
        xd = RealMatrix(undef, size(last_sol.xd))
        ud = RealMatrix(undef, size(last_sol.ud))
        p = RealVector(undef, size(last_sol.p))
        xc = missing
        uc = missing
        cost = Inf
    else
        # SCP solved the problem!
        status = @sprintf "%s" SCP_SOLVED

        xd = last_sol.xd
        ud = last_sol.ud
        p = last_sol.p

        # >> Continuos-time trajectory <<
        # Since within-interval integration using Nsub points worked, using
        # twice as many this time around seems like a good heuristic
        Nc = 2*Nsub*(N-1)
        tc = RealVector(LinRange(0.0, 1.0, Nc))
        uc = Trajectory(td, ud, :linear)
        k = (t) -> max(floor(Int, t/(N-1))+1, N)
        F = (t, x) -> pbm.traj.f(t, k(t), x, sample(uc, t), p)
        xc_vals = rk4(F, last_sol.xd[:, 1], tc; full=true,
                      actions=pbm.traj.integ_actions)
        xc = Trajectory(tc, xc_vals, :linear)

        cost = last_sol.J_aug
    end

    sol = SCPSolution(status, algo, num_iters, cost, td, xd, ud, p, xc, uc)

    return sol
end # function

"""
    SCPHistory()

Empty history.

# Returns
- `history`: history with no entries.
"""
function SCPHistory()::SCPHistory
    subproblems = Vector{SCPSubproblem}(undef, 0)
    history = SCPHistory(subproblems)
    return history
end # function

"""
    correct_convex!(x_ref, u_ref, p_ref, pbm, constructor)

Find closest trajectory that satisfies the convex path constraints.

Closeness is measured in an L1-norm sense.

# Arguments
- `x_ref`: the discrete-time state trajectory to be projected.
- `u_ref`: the discrete-time input trajectory to be projected.
- `p_ref`: the parameter vector to be projected.
- `pbm`: the SCP problem definition.
- `constructor`: the subproblem constructor function (as a symbol).
"""
function correct_convex!(
    x_ref::RealMatrix,
    u_ref::RealMatrix,
    p_ref::RealVector,
    pbm::SCPProblem,
    constructor::Symbol)::Nothing

    # Parameters
    N = pbm.pars.N
    scale = pbm.common.scale

    # Initialize the problem
    opti = eval(Expr(:call, constructor, pbm))

    # Add the convex path constraints
    add_convex_state_constraints!(opti)
    add_convex_input_constraints!(opti)

    # Add epigraph constraints to make a convex cost for JuMP
    xh_ref = scale.iSx*(x_ref.-scale.cx)
    uh_ref = scale.iSu*(u_ref.-scale.cu)
    ph_ref = scale.iSp*(p_ref-scale.cp)
    dx = opti.xh-xh_ref
    du = opti.uh-uh_ref
    dp = opti.ph-ph_ref
    epi_x = @variable(opti.mdl, [1:N], base_name="τx")
    epi_u = @variable(opti.mdl, [1:N], base_name="τu")
    epi_p = @variable(opti.mdl, base_name="τp")
    C = CLP.ConvexCone
    add! = CLP.add!
    for k = 1:N
        add!(opti.mdl, C(vcat(epi_x[k], dx[:, k]), CLP.L1))
        add!(opti.mdl, C(vcat(epi_u[k], du[:, k]), CLP.L1))
    end
    add!(opti.mdl, C(vcat(epi_p, dp), CLP.L1))

    # Define the cost
    cost = sum(epi_x)+sum(epi_u)+epi_p
    set_objective_function(opti.mdl, cost)
    set_objective_sense(opti.mdl, MOI.MIN_SENSE)

    # Solve
    optimize!(opti.mdl)

    # Save solution
    status = termination_status(opti.mdl)
    if (status==MOI.OPTIMAL || status==MOI.ALMOST_OPTIMAL)
        x_ref .= value.(opti.x)
        u_ref .= value.(opti.u)
        p_ref .= value.(opti.p)
    else
        msg = string("Solver failed to find the closest initial guess ",
                     "that satisfies the convex constraints (%s)")
        err = SCPError(0, SCP_GUESS_PROJECTION_FAILED,
                       @eval @sprintf($msg, $status))
        throw(err)
    end

    return nothing
end # function

"""
    compute_scaling(pars, traj, t)

Compute the scaling matrices given the problem definition.

# Arguments
- `pars`: the SCP algorithm parameters.
- `traj`: the trajectory problem definition.
- `t`: normalized discrete-time grid.

# Returns
- `scale`: the scaling structure.
"""
function compute_scaling(
    pars::SCPParameters,
    traj::TrajectoryProblem,
    t::RealVector)::SCPScaling

    # Parameters
    nx = traj.nx
    nu = traj.nu
    np = traj.np
    solver = pars.solver
    solver_opts = pars.solver_opts
    zero_intvl_tol = sqrt(eps())
    add! = CLP.add!

    # Map varaibles to these scaled intervals
    intrvl_x = [0.0; 1.0]
    intrvl_u = [0.0; 1.0]
    intrvl_p = [0.0; 1.0]

    # >> Compute physical variable bounding boxes <<

    x_bbox = fill(1.0, nx, 2)
    u_bbox = fill(1.0, nu, 2)
    p_bbox = fill(1.0, np, 2)
    x_bbox[:, 1] .= 0.0
    u_bbox[:, 1] .= 0.0
    p_bbox[:, 1] .= 0.0

    defs = [Dict(:dim => nx,
                 :set => traj.X,
                 :setcall => (t, k, x, u, p) -> traj.X(t, k, x, p),
                 :bbox => x_bbox,
                 :advice => :xrg,
                 :cost => (x, u, p, i) -> x[i]),
            Dict(:dim => nu,
                 :set => traj.U,
                 :setcall => (t, k, x, u, p) -> traj.U(t, k, u, p),
                 :bbox => u_bbox,
                 :advice => :urg,
                 :cost => (x, u, p, i) -> u[i]),
            Dict(:dim => np,
                 :set => traj.X,
                 :setcall => (t, k, x, u, p) -> traj.X(t, k, x, p),
                 :bbox => p_bbox,
                 :advice => :prg,
                 :cost => (x, u, p, i) -> p[i]),
            Dict(:dim => np,
                 :set => traj.U,
                 :setcall => (t, k, x, u, p) -> traj.U(t, k, u, p),
                 :bbox => p_bbox,
                 :advice => :prg,
                 :cost => (x, u, p, i) -> p[i])]

    for def in defs
        for j = 1:2 # 1:min, 2:max
            for i = 1:def[:dim]
                if !isnothing(getfield(traj, def[:advice])[i])
                    # Take user scaling advice
                    def[:bbox][i, j] = getfield(traj, def[:advice])[i][j]
                else
                    # Initialize JuMP model
                    mdl = Model()
                    set_optimizer(mdl, solver.Optimizer)
                    for (key,val) in solver_opts
                        set_optimizer_attribute(mdl, key, val)
                    end
                    # Variables
                    x = @variable(mdl, [1:nx])
                    u = @variable(mdl, [1:nu])
                    p = @variable(mdl, [1:np])
                    # Constraints
                    if !isnothing(def[:set])
                        for k = 1:length(t)
                            add!(mdl, def[:setcall](t[k], k, x, u, p))
                        end
                    end
                    # Cost
                    set_objective_function(mdl, def[:cost](x, u, p, i))
                    set_objective_sense(mdl, (j==1) ? MOI.MIN_SENSE :
                                        MOI.MAX_SENSE)
                    # Solve
                    optimize!(mdl)
                    # Record the solution
                    status = termination_status(mdl)
                    if (status==MOI.OPTIMAL || status==MOI.ALMOST_OPTIMAL)
                        # Nominal case
                        def[:bbox][i, j] = objective_value(mdl)
                    elseif !(status == MOI.DUAL_INFEASIBLE ||
                             status == MOI.NUMERICAL_ERROR)
                        msg = "Solver failed during variable scaling (%s)"
                        err = SCPError(0, SCP_SCALING_FAILED,
                                       @eval @sprintf($msg, $status))
                        throw(err)
                    end
                end
            end
        end
    end

    # >> Compute scaling matrices and offset vectors <<
    wdth_x = intrvl_x[2]-intrvl_x[1]
    wdth_u = intrvl_u[2]-intrvl_u[1]
    wdth_p = intrvl_p[2]-intrvl_p[1]

    # State scaling terms
    x_min, x_max = x_bbox[:, 1], x_bbox[:, 2]
    diag_Sx = (x_max-x_min)/wdth_x
    diag_Sx[diag_Sx .< zero_intvl_tol] .= 1.0
    Sx = collect(Diagonal(diag_Sx))
    iSx = inv(Sx)
    cx = x_min-diag_Sx*intrvl_x[1]

    # Input scaling terms
    u_min, u_max = u_bbox[:, 1], u_bbox[:, 2]
    diag_Su = (u_max-u_min)/wdth_u
    diag_Su[diag_Su .< zero_intvl_tol] .= 1.0
    Su = collect(Diagonal(diag_Su))
    iSu = inv(Su)
    cu = u_min-diag_Su*intrvl_u[1]

    # Parameter scaling terms
    p_min, p_max = p_bbox[:, 1], p_bbox[:, 2]
    diag_Sp = (p_max-p_min)/wdth_p
    diag_Sp[diag_Sp .< zero_intvl_tol] .= 1.0
    Sp = collect(Diagonal(diag_Sp))
    iSp = inv(Sp)
    cp = p_min-diag_Sp*intrvl_p[1]

    scale = SCPScaling(Sx, cx, Su, cu, Sp, cp, iSx, iSu, iSp)

    return scale
end # function

"""
    derivs(t, V, k, pbm, ref)

Compute concatenanted time derivative vector for dynamics discretization.

# Arguments
- `t`: the time.
- `V`: the current concatenated vector.
- `k`: the discrete time grid interval.
- `pbm`: the SCP problem definition.
- `ref`: the reference trajectory.

# Returns
- `dVdt`: the time derivative of V.
"""
function derivs(t::RealTypes,
                V::RealVector,
                k::Int,
                pbm::SCPProblem,
                ref::SCPSubproblemSolution)::RealVector
    # Parameters
    nx = pbm.traj.nx
    t_span = pbm.common.t_grid[k:k+1]

    # Get current values
    idcs = pbm.common.id
    x = V[idcs.x]
    u = linterp(t, ref.ud[:, k:k+1], t_span)
    p = ref.p
    Phi = reshape(V[idcs.A], (nx, nx))
    σ_m = (t_span[2]-t)/(t_span[2]-t_span[1])
    σ_p = (t-t_span[1])/(t_span[2]-t_span[1])

    # Compute the state time derivative and local linearization
    f = pbm.traj.f(t, k, x, u, p)
    A = pbm.traj.A(t, k, x, u, p)
    B = pbm.traj.B(t, k, x, u, p)
    F = pbm.traj.F(t, k, x, u, p)
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
end # function

"""
    discretize!(ref, pbm)

Discrete linear time varying dynamics computation. Compute the discrete-time
update matrices for the linearized dynamics about a reference trajectory. As a
byproduct, this calculates the defects needed for the trust region update.

# Arguments
- `ref`: reference solution about which to discretize.
- `pbm`: the SCP problem definition.
"""
function discretize!(
    ref::T, pbm::SCPProblem)::Nothing where {T<:SCPSubproblemSolution}

    ref.dyn.timing = get_time()

    # Parameters
    traj = pbm.traj
    nx = traj.nx
    nu = traj.nu
    np = traj.np
    N = pbm.pars.N
    Nsub = pbm.pars.Nsub
    t = pbm.common.t_grid
    sz_E = size(pbm.common.E)
    iSx = pbm.common.scale.iSx

    # Initialization
    idcs = pbm.common.id
    V0 = zeros(idcs.length)
    V0[idcs.A] = vec(I(nx))
    ref.feas = true

    # Propagate individually over each discrete-time interval
    for k = 1:N-1
        # Reset the state initial condition
        V0[idcs.x] = ref.xd[:, k]

        # Integrate
        f = (t, V) -> derivs(t, V, k, pbm, ref)
        t_subgrid = RealVector(LinRange(t[k], t[k+1], Nsub))
        V = rk4(f, V0, t_subgrid; actions=traj.integ_actions)

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
        E_k = A_k*reshape(EV, sz_E)

        # Save the discrete-time update matrices
        ref.dyn.A[:, :, k] = A_k
        ref.dyn.Bm[:, :, k] = Bm_k
        ref.dyn.Bp[:, :, k] = Bp_k
        ref.dyn.F[:, :, k] = F_k
        ref.dyn.r[:, k] = r_k
        ref.dyn.E[:, :, k] = E_k

        # Take this opportunity to comput the defect, which will be needed
        # later for the trust region update
        x_next = ref.xd[:, k+1]
        ref.defect[:, k] = x_next-xV
        if norm(iSx*ref.defect[:, k], Inf) > pbm.pars.feas_tol
            ref.feas = false
        end

    end

    ref.dyn.timing = (get_time()-ref.dyn.timing)/1e9

    return nothing
end # function

"""
    add_dynamics!(spbm)

Add dynamics constraints to the problem.

# Arguments
- `spbm`: the subproblem definition.
"""
function add_dynamics!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    __prg = spbm.__prg
    __x = spbm.__x
    __u = spbm.__u
    __p = spbm.__p
    __vd = spbm.__vd

    # Add dynamics constraint to optimization model
    for k = 1:N-1
        __xk, __xkp1 = __x[:, k], __x[:, k+1]
        __uk, __ukp1, __vdk = __u[:, k], __u[:, k+1], __vd[:, k]
        A = spbm.ref.dyn.A[:, :, k]
        Bm = spbm.ref.dyn.Bm[:, :, k]
        Bp = spbm.ref.dyn.Bp[:, :, k]
        F = spbm.ref.dyn.F[:, :, k]
        r = spbm.ref.dyn.r[:, k]
        E = spbm.ref.dyn.E[:, :, k]
        @add_constraint(
            __prg, ZERO, "dynamics",
            (__xk, __xkp1, __uk, __ukp1, __p, __vdk),
            begin
                local xk, xkp1, uk, ukp1, p, vdk = arg #noerr
                xkp1-(A*xk+Bm*uk+Bp*ukp1+F*p+r+E*vdk)
            end)
    end

    return nothing
end # function

"""
    add_convex_state_constraints!(spbm)

Add convex state constraints.

# Arguments
- `spbm`: the subproblem definition.
"""
function add_convex_state_constraints!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    traj_pbm = spbm.def.traj
    t = spbm.def.common.t_grid
    __prg = spbm.__prg
    __x = spbm.__x
    __p = spbm.__p

    if !isnothing(traj_pbm.__X)
        for k = 1:N
            traj_pbm.__X(__prg, t[k], k, __x[:, k], __p)
        end
    end

    return nothing
end # function

"""
    add_convex_input_constraints!(spbm)

Add convex input constraints.

# Arguments
- `spbm`: the subproblem definition.
"""
function add_convex_input_constraints!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    traj_pbm = spbm.def.traj
    t = spbm.def.common.t_grid
    __prg = spbm.__prg
    __u = spbm.__u
    __p = spbm.__p

    if !isnothing(traj_pbm.__U)
        for k = 1:N
            traj_pbm.__U(__prg, t[k], k, __u[:, k], __p)
        end
    end

    return nothing
end # function

"""
    add_nonconvex_constraints!(spbm)

Add non-convex state, input, and parameter constraints.

# Arguments
- `spbm`: the subproblem definition.
"""
function add_nonconvex_constraints!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    traj_pbm = spbm.def.traj
    t = spbm.def.common.t_grid
    __prg = spbm.__prg
    __x = spbm.__x
    __u = spbm.__u
    __p = spbm.__p
    nx = traj_pbm.nx
    nu = traj_pbm.nu
    np = traj_pbm.np
    xb = spbm.ref.xd
    ub = spbm.ref.ud
    pb = spbm.ref.p

    # Problem-specific convex constraints
    for k = 1:N
        if !isnothing(traj_pbm.s)
            tkxup = (t[k], k, xb[:, k], ub[:, k], pb)
            s = traj_pbm.s(tkxup...)
            ns = length(s)
            if k==1
                spbm.__vs = @new_variable(__prg, (ns, N), "vs")
            end
            C = !isnothing(traj_pbm.C) ? traj_pbm.C(tkxup...) : zeros(ns, nx)
            D = !isnothing(traj_pbm.D) ? traj_pbm.D(tkxup...) : zeros(ns, nu)
            G = !isnothing(traj_pbm.G) ? traj_pbm.G(tkxup...) : zeros(ns, np)
            r = s-C*xb[:, k]-D*ub[:, k]-G*pb

            __xk, __uk, __vsk = __x[:,k], __u[:,k], spbm.__vs[:, k]

            @add_constraint(
                __prg, NONPOS, "path_ncvx",
                (__xk, __uk, __p, __vsk),
                # Value
                begin
                    local xk, uk, p, vsk = arg #noerr
                    local lhs = C*xk+D*uk+G*p+r
                    lhs-vsk
                end,
                # Jacobians
                begin
                    local xk, uk, p, vsk = arg #noerr
                    local dim = length(vsk)
                    local J = Dict()
                    J[1] = C
                    J[2] = D
                    J[3] = G
                    J[4] = -collect(Int, I(dim))
                    J
                end)
        else
            break
        end
    end

    return nothing
end # function

"""
    add_bcs!(spbm[; relaxed])

Add boundary condition constraints to the problem.

# Arguments
- `spbm`: the subproblem definition.

# Keywords
- `relaxed`: (optional) if true then relax equalities with a virtual control,
  else impose the linearized boundary conditions exactly.
"""
function add_bcs!(
    spbm::T; relaxed::Bool=true)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    traj = spbm.def.traj
    __prg = spbm.__prg
    __x0 = spbm.__x[:, 1]
    __xf = spbm.__x[:, end]
    __p = spbm.__p
    nx = traj.nx
    np = traj.np
    xb0 = spbm.ref.xd[:, 1]
    xbf = spbm.ref.xd[:, end]
    pb = spbm.ref.p

    # Initial condition
    if !isnothing(traj.gic)
        gic = traj.gic(xb0, pb)
        nic = length(gic)
        H0 = !isnothing(traj.H0) ? traj.H0(xb0, pb) : zeros(nic, nx)
        K0 = !isnothing(traj.K0) ? traj.K0(xb0, pb) : zeros(nic, np)
        ℓ0 = gic-H0*xb0-K0*pb

        if relaxed
            spbm.__vic = @new_variable(__prg, nic, "vic")
            @add_constraint(__prg, ZERO, "initial_condition",
                            (__x0, __p, spbm.__vic),
                            begin
                                local x0, p, vic = arg #noerr
                                local lhs = H0*x0+K0*p+ℓ0
                                lhs+vic
                            end)
        else
            @add_constraint(__prg, ZERO, "initial_condition",
                            (__x0, __p),
                            begin
                                local x0, p = arg #noerr
                                local lhs = H0*x0+K0*p+ℓ0
                                lhs
                            end)
        end
    end

    # Terminal condition
    if !isnothing(traj.gtc)
        gtc = traj.gtc(xbf, pb)
        ntc = length(gtc)
        Hf = !isnothing(traj.Hf) ? traj.Hf(xbf, pb) : zeros(ntc, nx)
        Kf = !isnothing(traj.Kf) ? traj.Kf(xbf, pb) : zeros(ntc, np)
        ℓf = gtc-Hf*xbf-Kf*pb

        if relaxed
            spbm.__vtc = @new_variable(__prg, ntc, "vtc")
            @add_constraint(__prg, ZERO, "terminal_condition",
                            (__xf, __p, spbm.__vtc),
                            begin
                                local xf, p, vtc = arg #noerr
                                local lhs = Hf*xf+Kf*p+ℓf
                                lhs+vtc
                            end)
        else
            @add_constraint(__prg, ZERO, "terminal_condition",
                            (__xf, __p),
                            begin
                                local xf, p = arg #noerr
                                local lhs = Hf*xf+Kf*p+ℓf
                                lhs
                            end)
        end
    end

    return nothing
end # function

"""
    solution_deviation(spbm)

Compute the deviation of subproblem solution from the reference. It is assumed
that the function received a solved subproblem.

# Arguments
- `spbm`: the subproblem structure.

# Returns
- `deviation`: a measure of deviation of `spbm.sol` from `spbm.ref`.
"""
function solution_deviation(spbm::T)::RealTypes where {T<:SCPSubproblem}
    # Extract values
    pbm = spbm.def
    N = pbm.pars.N
    q = pbm.pars.q_exit
    scale = pbm.common.scale
    ref = spbm.ref
    sol = spbm.sol
    xh = scale.iSx*(sol.xd.-scale.cx)
    ph = scale.iSp*(sol.p-scale.cp)
    xh_ref = scale.iSx*(ref.xd.-scale.cx)
    ph_ref = scale.iSp*(ref.p-scale.cp)

    # Compute deviation
    dp = norm(ph-ph_ref, q)
    dx = 0.0
    for k = 1:N
        dx = max(dx, norm(xh[:, k]-xh_ref[:, k], q))
    end
    deviation = dp+dx

    return deviation
end # function

"""
    solve_subproblem!(spbm)

Solve the SCP method's convex subproblem via numerical optimization.

# Arguments
- `spbm`: the subproblem structure.
"""
function solve_subproblem!(spbm::T)::Nothing where {T<:SCPSubproblem}
    # Optimize
    solve!(spbm.__prg)

    # Save the solution
    SCPSubproblemSolution!(spbm)

    return nothing
end # function

"""
    unsafe_solution(sol)

Check if the subproblem optimization had issues. A solution is judged unsafe if
the numerical optimizer exit code indicates that there were serious problems in
solving the subproblem.

# Arguments
- `sol`: the subproblem or directly its solution.

# Returns
- `unsafe`: true if the subproblem solution process "failed".
"""
function unsafe_solution(sol::Union{T, V})::Bool where {
    T<:SCPSubproblemSolution, V<:SCPSubproblem}

    # If the parent subproblem passed in, then get its solution
    if typeof(sol) <: SCPSubproblem
        sol = sol.sol
    end

    if !sol.unsafe
        safe = sol.status==MOI.OPTIMAL || sol.status==MOI.ALMOST_OPTIMAL
        sol.unsafe = !safe
    end

    return sol.unsafe
end # function

"""
    overhead!(spbm)

Compute solution time overhead introduced by the surrounding code.

# Arguments
- `spbm`: the subproblem structure.
"""
function overhead!(spbm::T)::Nothing where {T<:SCPSubproblem}
    useful_time = (spbm.timing[:discretize]+spbm.timing[:formulate]+
                   spbm.timing[:solve])
    spbm.timing[:total] = (get_time()-spbm.timing[:total])/1e9
    spbm.timing[:overhead] = spbm.timing[:total]-useful_time
    return nothing
end # function

"""
    save!(hist, spbm)

Add subproblem to SCP history.

# Arguments
- `hist`: the history.
- `spbm`: subproblem structure.
"""
function save!(hist::SCPHistory,
               spbm::SCPSubproblem)::Nothing
    spbm.timing[:formulate] = (get_time()-spbm.timing[:formulate])/1e9
    push!(hist.subproblems, spbm)
    return nothing
end # function

"""
    get_time()

The the current time in nanoseconds.
"""
function get_time()::Int
    return Int(time_ns())
end # function
