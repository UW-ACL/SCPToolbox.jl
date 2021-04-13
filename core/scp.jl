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

using Printf

include("../utils/types.jl")
include("../utils/helper.jl")
include("problem.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Abstract types :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

abstract type SCPParameters end
abstract type SCPSubproblem end
abstract type SCPSubproblemSolution end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Variable scaling parameters.

Holds the SCP subproblem internal scaling parameters, which makes the numerical
optimization subproblems better conditioned.
"""
struct SCPScaling
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

""" Indexing arrays for convenient access during dynamics discretization.

Container of indices useful for extracting variables from the propagation
vector during the linearized dynamics discretization process.
"""
struct SCPDiscretizationIndices
    x::T_IntRange  # Indices for state
    A::T_IntRange  # Indices for A matrix
    Bm::T_IntRange # Indices for B_{-} matrix
    Bp::T_IntRange # Indices for B_{+} matrix
    F::T_IntRange  # Indices for S matrix
    r::T_IntRange  # Indices for r vector
    E::T_IntRange  # Indices for E matrix
    length::T_Int  # Propagation vector total length
end

"""" Common constant terms used throughout the algorithm."""
struct SCPCommon
    # >> Discrete-time grid <<
    Δt::T_Real           # Discrete time step
    t_grid::T_RealVector # Grid of scaled timed on the [0,1] interval
    E::T_RealMatrix      # Continuous-time matrix for dynamics virtual control
    scale::SCPScaling    # Variable scaling
    id::SCPDiscretizationIndices # Convenience indices during propagation
    table::T_Table       # Iteration info table (printout to REPL)
end

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
            msg = string("ERROR: the current implementation only supports",
                         " problems with at least 1 state and 1 control.")
            err = SCPError(0, SCP_BAD_PROBLEM, msg)
            throw(err)
        end

        pbm = new{typeof(pars)}(pars, traj, common)

        return pbm
    end
end

""" Overall trajectory solution.

Structure which holds the trajectory solution that the SCP algorithm returns.
"""
struct SCPSolution
    # >> Properties <<
    status::T_String  # Solution status (success? failure?)
    algo::T_String    # Which algorithm was used to obtain this solution
    iterations::T_Int # Number of SCP iterations that occurred
    cost::T_Real      # The original convex cost function
    # >> Discrete-time trajectory <<
    td::T_RealVector  # Discrete times
    xd::T_RealMatrix  # States
    ud::T_RealMatrix  # Inputs
    p::T_RealVector   # Parameter vector
    # >> Continuous-time trajectory <<
    xc::Union{T_ContinuousTimeTrajectory, Missing} # States
    uc::Union{T_ContinuousTimeTrajectory, Missing} # Inputs
end

"""" SCP iteration history data."""
struct SCPHistory{T<:SCPSubproblem}
    subproblems::Vector{T} # Subproblems
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
    E::T_RealMatrix)::SCPDiscretizationIndices

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
end

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
    table::T_Table)::SCPProblem where {T<:SCPParameters}

    # Compute the common constant terms
    t_grid = T_RealVector(LinRange(0.0, 1.0, pars.N))
    Δt = t_grid[2]-t_grid[1]
    E = T_RealMatrix(I(traj.nx))
    scale = _scp__compute_scaling(pars, traj, t_grid)
    idcs = SCPDiscretizationIndices(traj, E)
    consts = SCPCommon(Δt, t_grid, E, scale, idcs, table)

    pbm = SCPProblem(pars, traj, consts)

    return pbm
end

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
    spbm.sol.status = termination_status(spbm.mdl)

    # Save statistics about the subproblem and its solution
    spbm.timing[:solve] = solve_time(spbm.mdl)
    spbm.timing[:discretize] = spbm.sol.dyn.timing
    spbm.nvar = num_variables(spbm.mdl)
    moi2sym = Dict("MathOptInterface.Zeros" => :zero,
                   "MathOptInterface.Nonpositives" => :nonpos,
                   "MathOptInterface.NormOneCone" => :l1,
                   "MathOptInterface.SecondOrderCone" => :soc,
                   "MathOptInterface.NormInfinityCone" => :linf,
                   "MathOptInterface.GeometricMeanCone" => :geom,
                   "MathOptInterface.ExponentialCone" => :exp)
    dim = (ref) -> MOI.dimension(moi_set(constraint_object(ref)))
    cons_types = list_of_constraint_types(spbm.mdl)
    for cons_type in cons_types
        function_type, set_type = cons_type
        key = moi2sym[string(set_type)]
        refs = all_constraints(spbm.mdl, function_type, set_type)
        if key in (:zero, :nonpos)
            spbm.ncons[key] = sum(dim(ref) for ref in refs)
        else
            spbm.ncons[key] = T_IntVector([dim(ref) for ref in refs])
        end
    end

    return nothing
end

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
    nx = pbm.traj.nx
    nu = pbm.traj.nu
    np = pbm.traj.np
    td = pbm.common.t_grid
    algo = last_spbm.algo

    if _scp__unsafe_solution(last_sol)
        # SCP failed :(
        status = @sprintf "%s (%s)" SCP_FAILED last_sol.status
        xd = T_RealMatrix(undef, size(last_sol.xd))
        ud = T_RealMatrix(undef, size(last_sol.ud))
        p = T_RealVector(undef, size(last_sol.p))
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
        tc = T_RealVector(LinRange(0.0, 1.0, Nc))
        uc = T_ContinuousTimeTrajectory(td, ud, :linear)
        k = (t) -> max(floor(T_Int, t/(N-1))+1, N)
        F = (t, x) -> pbm.traj.f(t, k(t), x, sample(uc, t), p)
        xc_vals = rk4(F, @first(last_sol.xd), tc; full=true,
                      actions=pbm.traj.integ_actions)
        xc = T_ContinuousTimeTrajectory(tc, xc_vals, :linear)

        cost = last_sol.J_aug
    end

    sol = SCPSolution(status, algo, num_iters, cost, td, xd, ud, p, xc, uc)

    return sol
end

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
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    _scp__compute_scaling(pars, traj, t)

Compute the scaling matrices given the problem definition.

# Arguments
- `pars`: the SCP algorithm parameters.
- `traj`: the trajectory problem definition.
- `t`: normalized discrete-time grid.

# Returns
- `scale`: the scaling structure.
"""
function _scp__compute_scaling(
    pars::T,
    traj::TrajectoryProblem,
    t::T_RealVector)::SCPScaling where {T<:SCPParameters}

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
                            add_conic_constraints!(
                                mdl, def[:setcall](@k(t), k, x, u, p))
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

    # Parameter scaling terms
    p_min, p_max = p_bbox[:, 1], p_bbox[:, 2]
    diag_Sp = (p_max-p_min)/wdth_p
    diag_Sp[diag_Sp .< zero_intvl_tol] .= 1.0
    Sp = Diagonal(diag_Sp)
    iSp = inv(Sp)
    cp = p_min-diag_Sp*intrvl_p[1]

    scale = SCPScaling(Sx, cx, Su, cu, Sp, cp, iSx, iSu, iSp)

    return scale
end

"""
    _scp__derivs(t, V, k, pbm, ref)

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
function _scp__derivs(t::T_Real,
                      V::T_RealVector,
                      k::T_Int,
                      pbm::SCPProblem,
                      ref::T)::T_RealVector where {
                          T<:SCPSubproblemSolution}
    # Parameters
    nx = pbm.traj.nx
    N = pbm.pars.N
    t_span = @k(pbm.common.t_grid, k, k+1)

    # Get current values
    idcs = pbm.common.id
    x = V[idcs.x]
    u = linterp(t, @k(ref.ud, k, k+1), t_span)
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
end

"""
    _scp__discretize!(ref, pbm)

Discrete linear time varying dynamics computation. Compute the discrete-time
update matrices for the linearized dynamics about a reference trajectory. As a
byproduct, this calculates the defects needed for the trust region update.

# Arguments
- `ref`: reference solution about which to discretize.
- `pbm`: the SCP problem definition.
"""
function _scp__discretize!(
    ref::T, pbm::SCPProblem)::Nothing where {T<:SCPSubproblemSolution}

    ref.dyn.timing = time_ns()

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
        V0[idcs.x] = @k(ref.xd)

        # Integrate
        f = (t, V) -> _scp__derivs(t, V, k, pbm, ref)
        t_subgrid = T_RealVector(LinRange(@k(t), @kp1(t), Nsub))
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
        @k(ref.dyn.A) = A_k
        @k(ref.dyn.Bm) = Bm_k
        @k(ref.dyn.Bp) = Bp_k
        @k(ref.dyn.F) = F_k
        @k(ref.dyn.r) = r_k
        @k(ref.dyn.E) = E_k

        # Take this opportunity to comput the defect, which will be needed
        # later for the trust region update
        x_next = @kp1(ref.xd)
        @k(ref.defect) = x_next-xV
        if norm(iSx*@k(ref.defect), Inf) > pbm.pars.feas_tol
            ref.feas = false
        end

    end

    ref.dyn.timing = (time_ns()-ref.dyn.timing)/1e9

    return nothing
end

"""
    _scp__add_dynamics!(spbm)

Add dynamics constraints to the problem.

# Arguments
- `spbm`: the subproblem definition.
"""
function _scp__add_dynamics!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    x = spbm.x
    u = spbm.u
    p = spbm.p
    vd = spbm.vd

    # Add dynamics constraint to optimization model
    acc! = add_conic_constraint!
    Cone = T_ConvexConeConstraint
    for k = 1:N-1
        xk, xkp1, uk, ukp1, vdk = @k(x), @kp1(x), @k(u), @kp1(u), @k(vd)
        A = @k(spbm.ref.dyn.A)
        Bm = @k(spbm.ref.dyn.Bm)
        Bp = @k(spbm.ref.dyn.Bp)
        F = @k(spbm.ref.dyn.F)
        r = @k(spbm.ref.dyn.r)
        E = @k(spbm.ref.dyn.E)
        acc!(spbm.mdl, Cone(xkp1-(A*xk+Bm*uk+Bp*ukp1+F*p+r+E*vdk), :zero))
    end

    return nothing
end

"""
    _scp__add_convex_state_constraints!(spbm)

Add convex state constraints.

# Arguments
- `spbm`: the subproblem definition.
"""
function _scp__add_convex_state_constraints!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    traj_pbm = spbm.def.traj
    t = spbm.def.common.t_grid
    x = spbm.x
    p = spbm.p

    if !isnothing(traj_pbm.X)
        for k = 1:N
            xk_in_X = traj_pbm.X(@k(t), k, @k(x), p)
            correct_type = typeof(xk_in_X)<:(
                Vector{T} where {T<:T_ConvexConeConstraint})
            if !correct_type
                msg = string("ERROR: input constraint must be in conic form.")
                err = SCPError(k, SCP_BAD_ARGUMENT, msg)
                throw(err)
            end
            add_conic_constraints!(spbm.mdl, xk_in_X)
        end
    end

    return nothing
end

"""
    _scp__add_convex_input_constraints!(spbm)

Add convex input constraints.

# Arguments
- `spbm`: the subproblem definition.
"""
function _scp__add_convex_input_constraints!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    traj_pbm = spbm.def.traj
    t = spbm.def.common.t_grid
    u = spbm.u
    p = spbm.p

    if !isnothing(traj_pbm.U)
        for k = 1:N
            uk_in_U = traj_pbm.U(@k(t), k, @k(u), p)
            correct_type = typeof(uk_in_U)<:(
                Vector{T} where {T<:T_ConvexConeConstraint})
            if !correct_type
                msg = string("ERROR: input constraint must be in conic form.")
                err = SCPError(k, SCP_BAD_ARGUMENT, msg)
                throw(err)
            end
            add_conic_constraints!(spbm.mdl, uk_in_U)
        end
    end

    return nothing
end

"""
    _scp__add_nonconvex_constraints!(spbm)

Add non-convex state, input, and parameter constraints.

# Arguments
- `spbm`: the subproblem definition.
"""
function _scp__add_nonconvex_constraints!(
    spbm::T)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    N = spbm.def.pars.N
    traj_pbm = spbm.def.traj
    t = spbm.def.common.t_grid
    nx = traj_pbm.nx
    nu = traj_pbm.nu
    np = traj_pbm.np
    xb = spbm.ref.xd
    ub = spbm.ref.ud
    pb = spbm.ref.p
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Problem-specific convex constraints
    acc! = add_conic_constraint!
    Cone = T_ConvexConeConstraint
    for k = 1:N
        if !isnothing(traj_pbm.s)
            tkxup = (@k(t), k, @k(xb), @k(ub), pb)
            s = traj_pbm.s(tkxup...)
            ns = length(s)
            C = !isnothing(traj_pbm.C) ? traj_pbm.C(tkxup...) : zeros(ns, nx)
            D = !isnothing(traj_pbm.D) ? traj_pbm.D(tkxup...) : zeros(ns, nu)
            G = !isnothing(traj_pbm.G) ? traj_pbm.G(tkxup...) : zeros(ns, np)
            r = s-C*@k(xb)-D*@k(ub)-G*pb
            lhs = C*@k(x)+D*@k(u)+G*p+r

            if k==1
                spbm.vs = @variable(spbm.mdl, [1:ns, 1:N], base_name="vs")
            end

            acc!(spbm.mdl, Cone(lhs-@k(spbm.vs), :nonpos))
        else
            spbm.vs = @variable(spbm.mdl, [1:0, 1:N], base_name="vs")
            break
        end
    end

    return nothing
end

"""
    _scp__original_cost(x, u, p, pbm)

Compute the original problem cost function.

# Arguments
- `x`: the discrete-time state trajectory.
- `u`: the discrete-time input trajectory.
- `p`: the parameter vector.
- `pbm`: the SCP problem definition.

# Returns
- `cost`: the original cost.
"""
function _scp__original_cost(
    x::T_OptiVarMatrix,
    u::T_OptiVarMatrix,
    p::T_OptiVarVector,
    pbm::SCPProblem)::T_Objective

    # Parameters
    N = pbm.pars.N
    t = pbm.common.t_grid
    traj_pbm = pbm.traj

    # Terminal cost
    xf = @last(x)
    J_term = isnothing(pbm.traj.φ) ? 0.0 : pbm.traj.φ(xf, p)

    # Integrated running cost
    J_run = Vector{T_Objective}(undef, N)
    for k = 1:N
        @k(J_run) = isnothing(pbm.traj.Γ) ? 0.0 : pbm.traj.Γ(@k(x), @k(u), p)
    end
    integ_J_run = trapz(J_run, t)

    cost = J_term+integ_J_run

    return cost
end

"""
    _scp__add_bcs!(spbm[; relaxed])

Add boundary condition constraints to the problem.

# Arguments
- `spbm`: the subproblem definition.

# Keywords
- `relaxed`: (optional) if true then relax equalities with a virtual control,
  else impose the linearized boundary conditions exactly.
"""
function _scp__add_bcs!(
    spbm::T; relaxed::T_Bool=true)::Nothing where {T<:SCPSubproblem}

    # Variables and parameters
    traj = spbm.def.traj
    nx = traj.nx
    np = traj.np
    x0 = @first(spbm.x)
    xb0 = @first(spbm.ref.xd)
    xf = @last(spbm.x)
    xbf = @last(spbm.ref.xd)
    p = spbm.p
    pb = spbm.ref.p

    # Initial condition
    acc! = add_conic_constraint!
    Cone = T_ConvexConeConstraint
    if !isnothing(traj.gic)
        gic = traj.gic(xb0, pb)
        nic = length(gic)
        H0 = !isnothing(traj.H0) ? traj.H0(xb0, pb) : zeros(nic, nx)
        K0 = !isnothing(traj.K0) ? traj.K0(xb0, pb) : zeros(nic, np)
        ℓ0 = gic-H0*xb0-K0*pb
        lhs = H0*x0+K0*p+ℓ0
        if relaxed
            spbm.vic = @variable(spbm.mdl, [1:nic], base_name="vic")
            acc!(spbm.mdl, Cone(lhs+spbm.vic, :zero))
        else
            acc!(spbm.mdl, Cone(lhs, :zero))
        end
    elseif relaxed
        spbm.vic = @variable(spbm.mdl, [1:0], base_name="vic")
    end

    # Terminal condition
    if !isnothing(traj.gtc)
        gtc = traj.gtc(xbf, pb)
        ntc = length(gtc)
        Hf = !isnothing(traj.Hf) ? traj.Hf(xbf, pb) : zeros(ntc, nx)
        Kf = !isnothing(traj.Kf) ? traj.Kf(xbf, pb) : zeros(ntc, np)
        ℓf = gtc-Hf*xbf-Kf*pb
        lhs = Hf*xf+Kf*p+ℓf
        if relaxed
            spbm.vtc = @variable(spbm.mdl, [1:ntc], base_name="vtc")
            acc!(spbm.mdl, Cone(lhs+spbm.vtc, :zero))
        else
            acc!(spbm.mdl, Cone(lhs, :zero))
        end
    elseif relaxed
        spbm.vtc = @variable(spbm.mdl, [1:0], base_name="vtc")
    end

    return nothing
end

"""
    _scp__solution_deviation(spbm)

Compute the deviation of subproblem solution from the reference. It is assumed
that the function received a solved subproblem.

# Arguments
- `spbm`: the subproblem structure.

# Returns
- `deviation`: a measure of deviation of `spbm.sol` from `spbm.ref`.
"""
function _scp__solution_deviation(spbm::T)::T_Real where {T<:SCPSubproblem}
    # Extract values
    pbm = spbm.def
    N = pbm.pars.N
    q = pbm.pars.q_exit
    scale = pbm.common.scale
    ref = spbm.ref
    sol = spbm.sol
    xh = scale.iSx*(sol.xd.-scale.cx)
    ph = scale.iSp*(sol.p-scale.cp)
    xh_ref = scale.iSx*(spbm.ref.xd.-scale.cx)
    ph_ref = scale.iSp*(spbm.ref.p-scale.cp)

    # Compute deviation
    dp = norm(ph-ph_ref, q)
    dx = 0.0
    for k = 1:N
        dx = max(dx, norm(@k(xh)-@k(xh_ref), q))
    end
    deviation = dp+dx

    return deviation
end

"""
    _scp__solve_subproblem!(spbm)

Solve the SCP method's convex subproblem via numerical optimization.

# Arguments
- `spbm`: the subproblem structure.
"""
function _scp__solve_subproblem!(spbm::T)::Nothing where {T<:SCPSubproblem}
    # Optimize
    optimize!(spbm.mdl)

    # Save the solution
    SCPSubproblemSolution!(spbm)

    return nothing
end

"""
    _scp__unsafe_solution(sol)

Check if the subproblem optimization had issues. A solution is judged unsafe if
the numerical optimizer exit code indicates that there were serious problems in
solving the subproblem.

# Arguments
- `sol`: the subproblem or directly its solution.

# Returns
- `unsafe`: true if the subproblem solution process "failed".
"""
function _scp__unsafe_solution(sol::Union{T, V})::T_Bool where {
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
end

"""
    _scp__overhead!(spbm)

Compute solution time overhead introduced by the surrounding code.

# Arguments
- `spbm`: the subproblem structure.
"""
function _scp__overhead!(spbm::T)::Nothing where {T<:SCPSubproblem}
    useful_time = (spbm.timing[:discretize]+spbm.timing[:formulate]+
                   spbm.timing[:solve])
    spbm.timing[:total] = (time_ns()-spbm.timing[:total])/1e9
    spbm.timing[:overhead] = spbm.timing[:total]-useful_time
    return nothing
end

"""
    _scp__save!(hist, spbm)

Add subproblem to SCP history.

# Arguments
- `hist`: the history.
- `spbm`: subproblem structure.
"""
function _scp__save!(hist::SCPHistory,
                     spbm::T)::Nothing where {T<:SCPSubproblem}
    spbm.timing[:formulate] = (time_ns()-spbm.timing[:formulate])/1e9
    push!(hist.subproblems, spbm)
    return nothing
end
