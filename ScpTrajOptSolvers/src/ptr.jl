#= PTR algorithm data structures and methods.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

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
    include("../../ScpTrajOptUtils/src/ScpTrajOptUtils.jl")
    include("../../ScpTrajOptParser/src/ScpTrajOptParser.jl")
    include("scp.jl")
    using .ScpTrajOptUtils
    using .ScpTrajOptParser
end

using LinearAlgebra
using JuMP
using Printf
using ScpTrajOptUtils
using ScpTrajOptParser

import ..ST, ..RealValue, ..IntRange, ..RealVector, ..RealMatrix, ..Trajectory,
    ..Objective, ..VariableVector, ..VariableMatrix

import ..SCPParameters, ..SCPSubproblem, ..SCPSubproblemSolution, ..SCPProblem,
    ..SCPSolution, ..SCPHistory

import ..discretize!, ..add_dynamics!, ..add_convex_state_constraints!,
    ..add_convex_input_constraints!, ..add_nonconvex_constraints!, ..add_bcs!,
    ..original_cost, ..solution_deviation, ..solve_subproblem!,
    ..unsafe_solution, ..overhead!, ..save!, ..get_time

const Variable = ST.Variable

export Parameters, create, solve

#= Structure holding the PTR algorithm parameters. =#
struct Parameters <: SCPParameters
    N::Int               # Number of temporal grid nodes
    Nsub::Int            # Number of subinterval integration time nodes
    iter_max::Int        # Maximum number of iterations
    wvc::RealValue       # Virtual control weight
    wtr::RealValue       # Trust region weight
    ε_abs::RealValue     # Absolute convergence tolerance
    ε_rel::RealValue     # Relative convergence tolerance
    feas_tol::RealValue  # Dynamic feasibility tolerance
    q_tr::RealValue      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::RealValue    # Stopping criterion norm
    solver::Module       # The numerical solver to use for the subproblems
    solver_opts::Dict{String, Any} # Numerical solver options
end # struct

#= PTR subproblem solution. =#
mutable struct SubproblemSolution <: SCPSubproblemSolution
    iter::Int             # PTR iteration number
    # >> Discrete-time rajectory <<
    xd::RealMatrix        # States
    ud::RealMatrix        # Inputs
    p::RealVector         # Parameter vector
    # >> Virtual control terms <<
    vd::RealMatrix        # Dynamics virtual control
    vs::RealMatrix        # Nonconvex constraints virtual control
    vic::RealVector       # Initial conditions virtual control
    vtc::RealVector       # Terminal conditions virtual control
    # >> Cost values <<
    J::RealValue          # The original cost
    J_tr::RealValue       # The trust region penalty
    J_vc::RealValue       # The virtual control penalty
    J_aug::RealValue      # Overall cost
    # >> Trajectory properties <<
    ηx::RealVector        # State trust region radii
    ηu::RealVector        # Input trust region radii
    ηp::RealValue         # Parameter trust region radii
    status::ST.ExitStatus # Numerical optimizer exit status
    feas::Bool            # Dynamic feasibility flag
    defect::RealMatrix    # "Defect" linearization accuracy metric
    deviation::RealValue  # Deviation from reference trajectory
    unsafe::Bool          # Indicator that the solution is unsafe to use
    dyn::ST.DLTV          # The dynamics
end # struct

#= Subproblem definition in JuMP format for the convex numerical optimizer. =#
mutable struct Subproblem <: SCPSubproblem
    iter::Int            # PTR iteration number
    mdl::Model           # The optimization problem handle
    algo::String         # SCP and convex algorithms used
    # >> Algorithm parameters <<
    def::SCPProblem      # The PTR problem definition
    # >> Reference and solution trajectories <<
    sol::Union{SubproblemSolution, Missing} # Solution trajectory
    ref::Union{SubproblemSolution, Missing} # Reference trajectory
    # >> Cost function <<
    J::Objective        # The original convex cost function
    J_tr::Objective     # The virtual control penalty
    J_vc::Objective     # The virtual control penalty
    J_aug::Objective    # Overall cost function
    # >> Scaled variables <<
    xh::VariableMatrix  # Discrete-time states
    uh::VariableMatrix  # Discrete-time inputs
    ph::VariableVector  # Parameter
    # >> Physical variables <<
    x::VariableMatrix   # Discrete-time states
    u::VariableMatrix   # Discrete-time inputs
    p::VariableVector   # Parameters
    # >> Virtual control (never scaled) <<
    vd::VariableMatrix  # Dynamics virtual control
    vs::VariableMatrix  # Nonconvex constraints virtual control
    vic::VariableVector # Initial conditions virtual control
    vtc::VariableVector # Terminal conditions virtual control
    # >> Trust region <<
    ηx::VariableVector  # State trust region radii
    ηu::VariableVector  # Input trust region radii
    ηp::Variable        # Parameter trust region radii
    # >> Statistics <<
    nvar::Int                       # Total number of decision variables
    ncons::Dict{Symbol, Any}        # Number of constraints
    timing::Dict{Symbol, RealValue} # Runtime profiling
end # struct

#= Construct the PTR problem definition.

Args:
    pars: PTR algorithm parameters.
    traj: the underlying trajectory optimization problem.

Returns:
    pbm: the problem structure ready for being solved by PTR. =#
function create(pars::Parameters,
                traj::TrajectoryProblem)::SCPProblem

    table = ST.Table([
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
end # function

#= Constructor for an empty convex optimization subproblem.

No cost or constraints. Just the decision variables and empty associated
parameters.

Args:
    pbm: the PTR problem being solved.
    iter: PTR iteration number.
    ref: (optional) the reference trajectory.

Returns:
    spbm: the subproblem structure. =#
function Subproblem(pbm::SCPProblem,
                       iter::Int,
                       ref::Union{SubproblemSolution,
                                  Missing}=missing)::Subproblem

    # Statistics
    timing = Dict(:formulate => get_time(), :total => get_time())
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
    vs = Matrix{Float64}(undef, 0, N)
    vic = Vector{Float64}(undef, 0)
    vtc = Vector{Float64}(undef, 0)

    # Trust region radii
    ηx = @variable(mdl, [1:N], base_name="ηx")
    ηu = @variable(mdl, [1:N], base_name="ηu")
    ηp = @variable(mdl, base_name="ηp")

    spbm = Subproblem(iter, mdl, algo, pbm, sol, ref, J, J_tr, J_vc,
                         J_aug, xh, uh, ph, x, u, p, vd, vs, vic, vtc,
                         ηx, ηu, ηp, nvar, ncons, timing)

    return spbm
end # function

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
function SubproblemSolution(
    x::RealMatrix,
    u::RealMatrix,
    p::RealVector,
    iter::Int,
    pbm::SCPProblem)::SubproblemSolution

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
    dyn = ST.DLTV(nx, nu, np, nv, N)

    vd = Matrix{Float64}(undef, 0, N)
    vs = Matrix{Float64}(undef, 0, N)
    vic = Vector{Float64}(undef, 0)
    vtc = Vector{Float64}(undef, 0)

    J = NaN
    J_tr = NaN
    J_vc = NaN
    J_aug = NaN
    J_aug = NaN

    subsol = SubproblemSolution(iter, x, u, p, vd, vs, vic, vtc, J,
                                   J_tr, J_vc, J_aug, ηx, ηu, ηp, status,
                                   feas, defect, deviation, unsafe, dyn)

    # Compute the DLTV dynamics around this solution
    discretize!(subsol, pbm)

    return subsol
end # function

#= Construct subproblem solution from a subproblem object.

Expects that the subproblem argument is a solved subproblem (i.e. one to which
numerical optimization has been applied).

Args:
    spbm: the subproblem structure.

Returns:
    sol: subproblem solution. =#
function SubproblemSolution(spbm::Subproblem)::SubproblemSolution
    # Extract the discrete-time trajectory
    x = value.(spbm.x)
    u = value.(spbm.u)
    p = value.(spbm.p)

    # Form the partly uninitialized subproblem
    sol = SubproblemSolution(x, u, p, spbm.iter, spbm.def)

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
end # function

"""
    ptr_solve(pbm[, hot])

Solve the optimal control problem using the penalized trust region (PTR)
method.

# Arguments
- `pbm`: the trajectory problem to be solved.
- `warm`: (optional) warm start solution.

# Returns
- `sol`: the PTR solution structur.
- `history`: PTR iteration data history.
"""
function solve(pbm::SCPProblem,
               warm::Union{Nothing, SCPSolution}=nothing)::Tuple{
                   Union{SCPSolution, Nothing},
                   SCPHistory}
    # ..:: Initialize ::..

    if isnothing(warm)
        ref = generate_initial_guess(pbm)
    else
        ref = warm_start(pbm, warm)
    end

    history = SCPHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        spbm = Subproblem(pbm, k, ref)

        add_dynamics!(spbm)
        add_convex_state_constraints!(spbm)
        add_convex_input_constraints!(spbm)
        add_nonconvex_constraints!(spbm)
        add_bcs!(spbm)
        add_trust_region!(spbm)
        add_cost!(spbm)

        save!(history, spbm)

        try
            # >> Solve the subproblem <<
            solve_subproblem!(spbm)

            # "Emergency exit" the PTR loop if something bad happened
            # (e.g. numerical problems)
            if unsafe_solution(spbm)
                print_info(spbm)
                break
            end

            # >> Check stopping criterion <<
            stop = check_stopping_criterion!(spbm)
            if stop
                print_info(spbm)
                break
            end

            # >> Update reference trajectory <<
            ref = spbm.sol
        catch e
            isa(e, SCPError) || rethrow(e)
            print_info(spbm, e)
            break
        end

        # >> Print iteration info <<
        print_info(spbm)
    end

    reset(pbm.common.table)

    # ..:: Save solution ::..
    sol = SCPSolution(history)

    return sol, history
end # function

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the initial trajectory guess.

Construct the initial trajectory guess. Calls problem-specific initial guess
generator, which is converted to an SubproblemSolution structure.

Args:
    pbm: the PTR problem structure.

Returns:
    guess: the initial guess. =#
function generate_initial_guess(
    pbm::SCPProblem)::SubproblemSolution

    # Construct the raw trajectory
    x, u, p = pbm.traj.guess(pbm.pars.N)
    guess = SubproblemSolution(x, u, p, 0, pbm)

    return guess
end # function

"""
    warm_start(pbm, warm)

Create initial guess from a warm start solution.

# Arguments
- `pbm`: the PTR problem structure.
- `warm`: warm start solution.

# Returns
- `guess`: the initial guess for PTR.
"""
function warm_start(pbm::SCPProblem,
                          warm::SCPSolution)::SubproblemSolution

    # Extract the warm-start trajectory
    x, u, p = warm.xd, warm.ud, warm.p
    guess = SubproblemSolution(x, u, p, 0, pbm)

    return guess
end # function

#= Add trust region constraint to the subproblem.

Args:
    spbm: the subproblem definition. =#
function add_trust_region!(spbm::Subproblem)::Nothing

    # Variables and parameters
    N = spbm.def.pars.N
    q = spbm.def.pars.q_tr
    scale = spbm.def.common.scale
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
    C = ConvexCone

    # Parameter trust region
    dp_lq = @variable(spbm.mdl, base_name="dp_lq")
    add!(spbm.mdl, C(vcat(dp_lq, dp), cone))
    if q==4
        wp = @variable(spbm.mdl, base_name="wp")
        add!(spbm.mdl, C(vcat(wp, dp_lq), :soc))
        add!(spbm.mdl, C(vcat(wp, ηp, 1), :geom))
    else
        add!(spbm.mdl, C(dp_lq-ηp, :nonpos))
    end

    # State and input trust regions
    dx_lq = @variable(spbm.mdl, [1:N], base_name="dx_lq")
    du_lq = @variable(spbm.mdl, [1:N], base_name="du_lq")
    for k = 1:N
        add!(spbm.mdl, C(vcat(dx_lq[k], dx[:, k]), cone))
        add!(spbm.mdl, C(vcat(du_lq[k], du[:, k]), cone))
        if q==4
            # State
            wx = @variable(spbm.mdl, base_name="wx")
            add!(spbm.mdl, C(vcat(wx, dx_lq[k]), :soc))
            add!(spbm.mdl, C(vcat(wx, ηx[k], 1), :geom))
            # Input
            wu = @variable(spbm.mdl, base_name="wu")
            add!(spbm.mdl, C(vcat(wu, du_lq[k]), :soc))
            add!(spbm.mdl, C(vcat(wu, ηu[k], 1), :geom))
        else
            # State
            add!(spbm.mdl, C(dx_lq[k]-ηx[k], :nonpos))
            # Input
            add!(spbm.mdl, C(du_lq[k]-ηu[k], :nonpos))
        end
    end

    return nothing
end # function

#= Define the subproblem cost function.

Args:
    spbm: the subproblem definition. =#
function add_cost!(spbm::Subproblem)::Nothing

    # Variables and parameters
    x = spbm.x
    u = spbm.u
    p = spbm.p

    # Compute the cost components
    spbm.J = original_cost(x, u, p, spbm.def)
    compute_trust_region_penalty!(spbm)
    compute_virtual_control_penalty!(spbm)

    # Overall cost
    spbm.J_aug = spbm.J+spbm.J_tr+spbm.J_vc

    # Associate cost function with the model
    set_objective_function(spbm.mdl, spbm.J_aug)
    set_objective_sense(spbm.mdl, MOI.MIN_SENSE)

    return nothing
end # function

#= Compute the subproblem cost trust region penalty term.

Args:
    spbm: the subproblem definition. =#
function compute_trust_region_penalty!(spbm::Subproblem)::Nothing

    # Variables and parameters
    t = spbm.def.common.t_grid
    wtr = spbm.def.pars.wtr
    ηx = spbm.ηx
    ηu = spbm.ηu
    ηp = spbm.ηp

    spbm.J_tr = wtr*(trapz(ηx, t)+trapz(ηu, t)+ηp)

    return nothing
end # function

#= Compute the subproblem cost virtual control penalty term.

Args:
    spbm: the subproblem definition. =#
function compute_virtual_control_penalty!(spbm::Subproblem)::Nothing

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
    C = ConvexCone
    P = @variable(spbm.mdl, [1:N], base_name="P")
    Pf = @variable(spbm.mdl, [1:2], base_name="Pf")
    for k = 1:N
        if k<N
            tmp = vcat(P[k], E[:, :, k]*vd[:, k], vs[:, k])
        else
            tmp = vcat(P[k], vs[:, k])
        end
        add!(spbm.mdl, C(tmp, :l1))
    end
    add!(spbm.mdl, C(vcat(Pf[1], vic), :l1))
    add!(spbm.mdl, C(vcat(Pf[2], vtc), :l1))
    spbm.J_vc = wvc*(trapz(P, t)+sum(Pf))

    return nothing
end # function

#= Check if stopping criterion is triggered.

Args:
    spbm: the subproblem definition.

Returns:
    stop: true if stopping criterion holds. =#
function check_stopping_criterion!(spbm::Subproblem)::Bool

    # Extract values
    pbm = spbm.def
    ref = spbm.ref
    sol = spbm.sol
    ε_abs = pbm.pars.ε_abs
    ε_rel = pbm.pars.ε_rel

    # Compute solution deviation from reference
    sol.deviation = solution_deviation(spbm)

    # Check predicted cost improvement
    J_ref = ref.J_aug
    J_sol = sol.J_aug
    improv_rel = abs(J_ref-J_sol)/abs(J_ref)

    # Compute stopping criterion
    stop = (spbm.iter>1 &&
            (sol.feas && (improv_rel<=ε_rel || sol.deviation<=ε_abs)))

    return stop
end # function

#= Print command line info message.

Args:
    spbm: the subproblem that was solved.
    err: an PTR-specific error message. =#
function print_info(spbm::Subproblem,
                          err::Union{Nothing, SCPError}=nothing)::Nothing

    # Convenience variables
    sol = spbm.sol
    ref = spbm.ref
    table = spbm.def.common.table

    if !isnothing(err)
        @printf "ERROR: %s, exiting\n" err.msg
    elseif unsafe_solution(sol)
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
        ΔJ = improvement_percent(sol.J_aug, ref.J_aug) #noerr
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

    overhead!(spbm)

    return nothing
end # function
