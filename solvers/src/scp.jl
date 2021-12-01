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

using LinearAlgebra
using JuMP
using Printf

export SCPSolution, SCPHistory

# ..:: Globals  ::..

const CLP = ConicLinearProgram

abstract type SCPParameters end
abstract type SCPSubproblem end

# ..:: Data structures ::..

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

"""" Common constant terms used throughout the algorithm."""
struct SCPCommon
    # >> Discrete-time grid <<
    Δt::RealTypes        # Discrete time step
    t_grid::RealVector   # Grid of scaled timed on the [0,1] interval
    E::RealMatrix        # Continuous-time matrix for dynamics virtual control
    scale::SCPScaling    # Variable scaling
    id::DiscretizationIndices # Convenience indices during propagation
    table::ST.Table      # Iteration info table (printout to REPL)
end # struct

""" Structure which contains all the necessary information to run SCP."""
struct SCPProblem{T<:SCPParameters} <: AbstractSCPProblem
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
    end
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
    idcs = DiscretizationIndices(traj, E, pars.disc_method)
    consts = SCPCommon(Δt, t_grid, E, scale, idcs, table)

    pbm = SCPProblem(pars, traj, consts)

    return pbm
end

"""
    SCPSubproblemSolution(spbm)

Create the subproblem solution structure. This calls the SCP algorithm-specific
function, and also saves some general properties of the solution.

# Arguments
- `spbm`: the subproblem structure.
"""
function SCPSubproblemSolution(spbm::T)::Nothing where {T<:SCPSubproblem}
    # Save the solution
    # (this complicated-looking thing calls the constructor for the
    #  SCPSubproblemSolution child type)
    constructor = Meta.parse(string(typeof(spbm.ref)))
    spbm.sol = eval(Expr(:call, constructor, spbm))

    # Save common solution properties
    spbm.sol.status = termination_status(spbm.prg)

    # Save statistics about the subproblem and its solution
    spbm.timing[:solve] = solve_time(spbm.prg)
    spbm.timing[:discretize] = spbm.sol.dyn.timing

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
    method = pbm.pars.disc_method
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
        xc = propagate(last_sol, pbm; res=Nc)
        if method==FOH
            uc = Trajectory(td, ud, :linear)
        elseif method==IMPULSE
            uc = Trajectory(td, ud, :impulse)
        end

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
end

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
                 :setcall => (prg, t, k, x, u, p) -> traj.X(prg, t, k, x, p),
                 :bbox => x_bbox,
                 :advice => :xrg,
                 :cost => (x, u, p, i) -> x[i]),
            Dict(:dim => nu,
                 :set => traj.U,
                 :setcall => (prg, t, k, x, u, p) -> traj.U(prg, t, k, u, p),
                 :bbox => u_bbox,
                 :advice => :urg,
                 :cost => (x, u, p, i) -> u[i]),
            Dict(:dim => np,
                 :set => traj.X,
                 :setcall => (prg, t, k, x, u, p) -> traj.X(prg, t, k, x, p),
                 :bbox => p_bbox,
                 :advice => :prg,
                 :cost => (x, u, p, i) -> p[i]),
            Dict(:dim => np,
                 :set => traj.U,
                 :setcall => (prg, t, k, x, u, p) -> traj.U(prg, t, k, u, p),
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
                    prg = ConicProgram(traj; solver=solver.Optimizer,
                                       solver_options=solver_opts)
                    # Variables
                    x = @new_variable(prg, nx, "x")
                    u = @new_variable(prg, nu, "u")
                    p = @new_variable(prg, np, "p")
                    # Constraints
                    if !isnothing(def[:set])
                        for k = 1:length(t)
                            def[:setcall](prg, t[k], k, x, u, p)
                        end
                    end
                    # Cost
                    minimize_cost = (j==1) ? 1 : -1
                    @add_cost(prg, (x, u, p), begin
                                  local x, u, p = arg
                                  minimize_cost*def[:cost](x, u, p, i)
                              end)
                    # Solve
                    status = solve!(prg)
                    # Record the solution
                    if (status==MOI.OPTIMAL || status==MOI.ALMOST_OPTIMAL)
                        # Nominal case
                        def[:bbox][i, j] = minimize_cost*objective_value(prg)
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

    # Constant parameter scaling terms

    scale = SCPScaling(Sx, cx, Su, cu, Sp, cp, iSx, iSu, iSp)

    return scale
end

"""
    add_dynamics!(spbm[; relaxed])

Add dynamics constraints to the problem.

# Arguments
- `spbm`: the subproblem definition.

# Keywords
- `relaxed`: (optional) if true then relax dynamics with a virtual control,
  else impose the linearized dynamics as-is.
"""
function add_dynamics!(spbm::SCPSubproblem;
                       relaxed::Bool=true)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    prg = spbm.prg
    dyn = spbm.ref.dyn
    x = spbm.x
    u = spbm.u
    p = spbm.p
    vd = relaxed ? spbm.vd : nothing

    # Add dynamics constraint to optimization model
    for k = 1:N-1
        state_update!(k, x, u, p, vd, dyn, prg)
    end

    return nothing
end

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
    prg = spbm.prg
    x = spbm.x
    p = spbm.p

    if !isnothing(traj_pbm.X)
        for k = 1:N
            traj_pbm.X(prg, t[k], k, x[:, k], p)
        end
    end

    return nothing
end

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
    prg = spbm.prg
    u = spbm.u
    p = spbm.p

    if !isnothing(traj_pbm.U)
        for k = 1:N
            traj_pbm.U(prg, t[k], k, u[:, k], p)
        end
    end

    return nothing
end

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
    prg = spbm.prg
    x = spbm.x
    u = spbm.u
    p = spbm.p
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
                spbm.vs = @new_variable(prg, (ns, N), "vs")
            end
            C = !isnothing(traj_pbm.C) ? traj_pbm.C(tkxup...) : zeros(ns, nx)
            D = !isnothing(traj_pbm.D) ? traj_pbm.D(tkxup...) : zeros(ns, nu)
            G = !isnothing(traj_pbm.G) ? traj_pbm.G(tkxup...) : zeros(ns, np)
            r = s-C*xb[:, k]-D*ub[:, k]-G*pb

            xk, uk, vsk = x[:,k], u[:,k], spbm.vs[:, k]

            @add_constraint(prg, NONPOS, "path_ncvx",
                            (xk, uk, p, vsk), begin
                                local xk, uk, p, vsk = arg
                                local lhs = C*xk+D*uk+G*p+r
                                lhs-vsk
                            end)
        else
            break
        end
    end

    return nothing
end

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
    prg = spbm.prg
    x0 = spbm.x[:, 1]
    xf = spbm.x[:, end]
    p = spbm.p
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
            spbm.vic = @new_variable(prg, nic, "vic")
            @add_constraint(prg, ZERO, "initial_condition",
                            (x0, p, spbm.vic), begin
                                local x0, p, vic = arg
                                local lhs = H0*x0+K0*p+ℓ0
                                lhs+vic
                            end)
        else
            @add_constraint(prg, ZERO, "initial_condition",
                            (x0, p), begin
                                local x0, p = arg
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
            spbm.vtc = @new_variable(prg, ntc, "vtc")
            @add_constraint(prg, ZERO, "terminal_condition",
                            (xf, p, spbm.vtc), begin
                                local xf, p, vtc = arg
                                local lhs = Hf*xf+Kf*p+ℓf
                                lhs+vtc
                            end)
        else
            @add_constraint(prg, ZERO, "terminal_condition",
                            (xf, p), begin
                                local xf, p = arg
                                local lhs = Hf*xf+Kf*p+ℓf
                                lhs
                            end)
        end
    end

    return nothing
end

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
end

"""
    solve_subproblem!(spbm)

Solve the SCP method's convex subproblem via numerical optimization.

# Arguments
- `spbm`: the subproblem structure.
"""
function solve_subproblem!(spbm::T)::Nothing where {T<:SCPSubproblem}
    # Optimize
    solve!(spbm.prg)

    # Save the solution
    SCPSubproblemSolution(spbm)

    return nothing
end

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
end

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
end

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
end
