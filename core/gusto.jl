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
    ω::T_Int          # Dynamics virtual control weight
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
    pen::T_Symbol     # Penalty type (:quad, :logsumexp)
    q_tr::T_Real      # Trust region norm (possible: 1, 2, 4 (2^2), Inf)
    q_exit::T_Real    # Stopping criterion norm
    solver::Module    # The numerical solver to use for the subproblems
    solver_opts::Dict{T_String, Any} # Numerical solver options
end

#= GuSTO subproblem solution. =#
mutable struct GuSTOSubproblemSolution <: SCPSubproblemSolution
    iter::T_Int          # SCvx iteration number
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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Apply the SCvx algorithm to solve the trajectory generation problem.

Args:
    pbm: the trajectory problem to be solved.

Returns:
    sol: the SCvx solution structure.
    history: SCvx iteration data history. =#
function gusto_solve(pbm::SCPProblem)::Tuple{Union{SCPSolution, Nothing},
                                             SCPHistory}
    # ..:: Initialize ::..

    λ = pbm.pars.λ_init
    η = pbm.pars.η_init
    # ref = _gusto__generate_initial_guess(pbm)

    history = SCPHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # TODO
    end

    # reset(pbm.common.table)

    # ..:: Save solution ::..
    sol = nothing # TODO SCPSolution(history)

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
# function _gusto__generate_initial_guess(
#     pbm::SCPProblem)::GuSTOSubproblemSolution

#     # Construct the raw trajectory
#     x, u, p = pbm.traj.guess(pbm.pars.N)
#     _scp__correct_convex!(x, u, pbm)
#     guess = GuSTOSubproblemSolution(x, u, p, 0, pbm)

#     # Compute the nonlinear cost
#     _scp__discretize!(guess, pbm)
#     _gusto__solution_cost!(guess, :nonlinear, pbm)

#     return guess
# end
