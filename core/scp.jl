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

#= Variable scaling parameters.

Holds the SCP subproblem internal scaling parameters, which makes the numerical
optimization subproblems better conditioned. =#
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

#= Indexing arrays for convenient access during dynamics discretization.

Container of indices useful for extracting variables from the propagation
vector during the linearized dynamics discretization process. =#
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

#= Common constant terms used throughout the algorithm. =#
struct SCPCommon
    # >> Discrete-time grid <<
    Δτ::T_Real           # Discrete time step
    τ_grid::T_RealVector # Grid of scaled timed on the [0,1] interval
    E::T_RealMatrix      # Continuous-time matrix for dynamics virtual control
    scale::SCPScaling    # Variable scaling
    id::SCPDiscretizationIndices # Convenience indices during propagation
    table::T_Table       # Iteration info table (printout to REPL)
end

#= Structure which contains all the necessary information to run SCP. =#
struct SCPProblem{T}
    pars::T                 # Algorithm parameters
    traj::TrajectoryProblem # The underlying trajectory problem
    common::SCPCommon       # Common precomputed terms
end

#= Overall trajectory solution.

Structure which holds the trajectory solution that the SCP algorithm
returns. =#
struct SCPSolution
    # >> Properties <<
    status::T_String  # Solution status (success? failure?)
    iterations::T_Int # Number of SCP iterations that occurred
    cost::T_Real      # The original convex cost function
    # >> Discrete-time trajectory <<
    τd::T_RealVector  # Discrete times
    xd::T_RealMatrix  # States
    ud::T_RealMatrix  # Inputs
    p::T_RealVector   # Parameter vector
    # >> Continuous-time trajectory <<
    xc::Union{T_ContinuousTimeTrajectory, Missing} # States
    uc::Union{T_ContinuousTimeTrajectory, Missing} # Inputs
end

#= SCP iteration history data. =#
struct SCPHistory{T<:SCPSubproblem}
    subproblems::Vector{T} # Subproblems
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Indexing arrays from problem definition.

Args:
    traj: the trajectory problem definition.
    E: the dynamics virtual control coefficient matrix.

Returns:
    idcs: the indexing array structure. =#
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

#= Construct the SCP problem definition.

This internally also computes the scaling matrices used to improve subproblem
numerics.

Args:
    pars: GuSTO algorithm parameters.
    traj: the underlying trajectory optimization problem.
    table: the iteration progress table.

Returns:
    pbm: the problem structure ready for being solved by GuSTO. =#
function SCPProblem(
    pars::T,
    traj::TrajectoryProblem,
    table::T_Table)::SCPProblem where {T<:SCPParameters}

    # Compute the common constant terms
    τ_grid = LinRange(0.0, 1.0, pars.N)
    Δτ = τ_grid[2]-τ_grid[1]
    E = T_RealMatrix(I(traj.nx))
    scale = _scp__compute_scaling(pars, traj)
    idcs = SCPDiscretizationIndices(traj, E)
    consts = SCPCommon(Δτ, τ_grid, E, scale, idcs, table)

    pbm = SCPProblem(pars, traj, consts)

    return pbm
end

#= Convert subproblem solution to a final trajectory solution.

This is what the SCP algorithm returns in the end to the user.

Args:
    history: SCP iteration history.

Returns:
    sol: the trajectory solution. =#
function SCPSolution(history::SCPHistory)::SCPSolution

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

    if _scp__unsafe_solution(last_sol)
        # SCvx failed :(
        status = @sprintf "%s (%s)" SCP_FAILED last_sol.status
        xd = T_RealMatrix(undef, size(last_sol.xd))
        ud = T_RealMatrix(undef, size(last_sol.ud))
        p = T_RealVector(undef, size(last_sol.p))
        xc = missing
        uc = missing
        cost = Inf
    else
        # SCvx solved the problem!
        status = @sprintf "%s" SCP_SOLVED

        xd = last_sol.xd
        ud = last_sol.ud
        p = last_sol.p

        # >> Continuos-time trajectory <<
        # Since within-interval integration using Nsub points worked, using
        # twice as many this time around seems like a good heuristic
        Nc = 2*Nsub*(N-1)
        τc = T_RealVector(LinRange(0.0, 1.0, Nc))
        uc = T_ContinuousTimeTrajectory(τd, ud, :linear)
        F = (τ, x) -> pbm.traj.f(x, sample(uc, τ), p)
        xc_vals = rk4(F, @first(last_sol.xd), τc; full=true,
                      actions=pbm.traj.integ_actions)
        xc = T_ContinuousTimeTrajectory(τc, xc_vals, :linear)

        cost = last_sol.L
    end

    sol = SCPSolution(status, num_iters, cost, τd, xd, ud, p, xc, uc)

    return sol
end

#= Empty history.

Returns:
    history: history with no entries. =#
function SCPHistory()::SCPHistory
    subproblems = Vector{SCPSubproblem}(undef, 0)
    history = SCPHistory(subproblems)
    return history
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the scaling matrices given the problem definition.

Args:
    pars: the SCP algorithm parameters.
    traj: the trajectory problem definition.

Returns:
    scale: the scaling structure. =#
function _scp__compute_scaling(
    pars::T,
    traj::TrajectoryProblem)::SCPScaling where {T<:SCPParameters}

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

    # >> Compute bounding boxes for state and input <<

    x_bbox = fill(1.0, nx, 2)
    u_bbox = fill(1.0, nu, 2)
    x_bbox[:, 1] .= 0.0
    u_bbox[:, 1] .= 0.0

    defs = [Dict(:dim => nx, :set => traj.X,
                 :bbox => x_bbox, :advice => :xrg),
            Dict(:dim => nu, :set => traj.U,
                 :bbox => u_bbox, :advice => :urg)]

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
                par = @variable(mdl, [1:np])
                # Constraints
                if !isnothing(def[:set])
                    add_conic_constraints!(mdl, def[:set](var, par))
                end
                # Cost
                set_objective_function(mdl, var[i])
                set_objective_sense(mdl, (j==1) ? MOI.MIN_SENSE :
                                    MOI.MAX_SENSE)
                # Solve
                optimize!(mdl)
                # Record the solution
                status = termination_status(mdl)
                if (status == MOI.DUAL_INFEASIBLE ||
                    status == MOI.NUMERICAL_ERROR)
                    if !isnothing(getfield(traj, def[:advice])[i])
                        # Take user scaling advice
                        def[:bbox][i, j] = getfield(traj, def[:advice])[i][j]
                    end
                else
                    if (status==MOI.OPTIMAL || status==MOI.ALMOST_OPTIMAL)
                        def[:bbox][i, j] = objective_value(mdl)
                    else
                        msg = "Solver failed during variable scaling (%s)"
                        err = SCPError(0, SCP_SCALING_FAILED,
                                       @eval @sprintf($msg, $status))
                        throw(err)
                    end
                end
            end
        end
    end

    # >> Compute bounding box for parameters <<

    p_bbox = fill(1.0, np, 2)
    p_bbox[:, 1] .= 0.0

    for j = 1:2 # 1:min, 2:max
        for i = 1:np
            if !isnothing(traj.prg[i])
                # Take user scaling advice
                p_bbox[i, j] = traj.prg[i][j]
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

#= Check if the subproblem optimization had issues.

A solution is judged unsafe if the numerical optimizer exit code indicates that
there were serious problems in solving the subproblem.

Args:
    sol: the subproblem or directly its solution.

Returns:
    unsafe: true if the subproblem solution process "failed". =#
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

#= Add subproblem to SCP history.

Args:
    hist: the history.
    spbm: subproblem structure. =#
function _scp__save!(hist::SCPHistory,
                     spbm::T)::Nothing where {T<:SCPSubproblem}
    push!(hist.subproblems, spbm)
    return nothing
end
