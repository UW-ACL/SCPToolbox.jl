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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Structure holding the GuSTO algorithm parameters. =#
struct GuSTOParameters
    N::T_Int          # Number of temporal grid nodes
    Nsub::T_Int       # Number of subinterval integration time nodes
    iter_max::T_Int   # Maximum number of iterations
    λ_init::T_Real    # Initial soft penalty weight
    λ_max::T_Real     # Maximum soft penalty weight
    ρ_0::T_Real       # Trust region update threshold (lower, good solution)
    ρ_1::T_Real       # Trust region update threshold (upper, bad solution)
    β_sh::T_Real      # Trust region shrinkage factor
    β_gr::T_Real      # Trust region growth factor
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
