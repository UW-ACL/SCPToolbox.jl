#= Optimization problem cost.

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
    include("general.jl")
    include("constraint.jl")
end

# ..:: Data structures ::..

"""
`QuadraticCost` stores the objective function of the problem. The goal is to
minimize this function. The function can be at most quadratic, however for
robustness (in JuMP) it has been observed that it is best to reformulate the
problem (via epigraph form) such that this function is affine. """
struct QuadraticCost
    J::ProgramFunction              # The core function
    jump::Types.Variable            # JuMP function object
    prog::Ref{AbstractConicProgram} # The parent conic program

    """
        QuadraticCost(J)

    Basic constructor. This will also call the appropriate JuMP function to
    associate the cost with the JuMP optimization model object.

    # Arguments
    - `J`: at most a quadratic function, that is to be minimized.
    - `prog`: the parent conic program which this is to be the cost of.

    # Returns
    - `cost`: the newly created cost function object.
    """
    function QuadraticCost(J::ProgramFunction,
                           prog::AbstractConicProgram)::QuadraticCost

        # Create the underlying JuMP cost
        J_value = J()[1]
        set_objective_function(jump_model(prog), J_value)
        jump = objective_function(jump_model(prog))
        set_objective_sense(jump_model(prog), MOI.MIN_SENSE)

        cost = new(J, jump, prog)

        return cost
    end # function
end # struct

# ..:: Methods ::..

""" Get the actual underlying cost function. """
core_function(J::QuadraticCost)::ProgramFunction = J.J

""" Get the current objective function value. """
value(J::QuadraticCost) = value(core_function(J))[1]

"""
    feasibility_cost(pars, jacobians)

A dummy function which represents a feasibility cost.

# Arguments
- `args...`: all arguments are ignored.

# Returns
- `J`: a constant-zero function representing the cost of a feasibility
  optimization problem.
"""
function feasibility_cost(
    args...)::DifferentiableFunctionOutput #nowarn

    J = DifferentiableFunctionOutput(0.0)
    return J
end # function
