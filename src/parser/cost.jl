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

import JuMP: value

export QuadraticCost

# ..:: Data structures ::..

"""
`FunctionLinearCombination` stores a linear combination of differentiable
functions, which supports mathematical operations like value evaluation and
differentiation.
"""
mutable struct FunctionLinearCombination
    f::Vector{ProgramFunction} # The functions being combined
    a::Types.RealVector        # The combination coefficients

    """
        FunctionLinearCombination(f[, a])

    Basic constructor starting with just one function.

    # Returns
    - `f`: the first function.
    - `a`: (optional) the coefficient multiplying the first function.

    # Returns
    - `f_comb`: the linear combination object.
    """
    function FunctionLinearCombination(
        f::ProgramFunction,
        a::Types.RealTypes = 1.0,
    )::FunctionLinearCombination

        f_list = Vector{ProgramFunction}(undef, 1)
        a_list = Types.RealVector(undef, 1)

        f_list[1] = f
        a_list[1] = a

        f_comb = new(f_list, a_list)

        return f_comb
    end
end

"""
`QuadraticCost` stores the objective function of the problem. The goal is to
minimize this function. The function can be at most quadratic, however for
robustness (in JuMP) it has been observed that it is best to reformulate the
problem (via epigraph form) such that this function is affine. """
mutable struct QuadraticCost
    terms::FunctionLinearCombination # The core cost terms
    prog::Ref{AbstractConicProgram}  # The parent conic program

    """
        QuadraticCost(J, prog[, a][; noupdate])

    Basic constructor. This will also call the appropriate JuMP function to
    associate the cost with the JuMP optimization model object.

    # Arguments
    - `J`: at most a quadratic function, that is to be minimized.
    - `prog`: the parent conic program which this is to be the cost of.
    - `a`: (optional) multiplicative coefficient in a linear combination.

    # Keywords
    - `noupdate`: (optional) flag to not update the underlying JuMP model cost.

    # Returns
    - `cost`: the newly created cost function object.
    """
    function QuadraticCost(
        J::ProgramFunction,
        prog::AbstractConicProgram,
        a::Types.RealTypes = 1.0;
        noupdate::Bool = false,
    )::QuadraticCost

        # Initialize the function combination
        terms = FunctionLinearCombination(J, a)

        cost = new(terms, prog)

        if !noupdate
            update_jump_cost!(cost)
        end

        return cost
    end
end

# ..:: Methods ::..

"""
    add!(F, f[, a])

Add a new function to the linear combination.

# Arguments
- `F`: the existing linear combination object.
- `f`: the new function to be added.
- `a`: (optional) the coefficient multiplying this function. Default is 1.
"""
function add!(
    F::FunctionLinearCombination,
    f::ProgramFunction,
    a::Types.RealTypes = 1.0,
)::Nothing
    push!(F.f, f)
    push!(F.a, a)
    return nothing
end

""" Get the number of functions in the linear combination. """
Base.length(F::FunctionLinearCombination)::Int = length(F.f)

"""
    value(F[; scalar])

Get the value of the function linear combination.

# Arguments
- `F`: the existing linear combination object.
- `scalar`: (optional) whether to output a scalar value.

# Returns
- `val`: the function value.
"""
function value(F::FunctionLinearCombination; scalar::Bool = false)::FunctionValueOutputType
    val = 0.0
    for i = 1:length(F)
        a, f = F.a[i], value(F.f[i], scalar = scalar)
        val += a * f
    end
    return val
end

"""
    jacobian(F, i, key)

Get the Jacobian `key` of the i-th term of the linear combination.

# Arguments
- `F`: the quadratic cost function.
- `i`: which term of the cost function to use.
- `key`: key adress of the Jacobian.

# Returns
- `jac`: the Jacobian.
"""
function jacobian(
    F::FunctionLinearCombination,
    i::Int,
    key::JacobianKeys,
)::JacobianValueType
    jac = F.a[i] * jacobian(F.f[i], key)
    return jac
end

"""
    all_jacobians(F)

Get a list of all the defined Jacobians. The `i`th element of the output vector
is a dictionary of all Jacobians available for the `i`th term.

# Arguments
- `F`: the quadratic cost function.

# Returns
- `jacs`: the list of available Jacobians.
"""
function all_jacobians(F::FunctionLinearCombination)::Vector{JacobianDictType}
    jacs = Vector{JacobianDictType}(undef, length(F))
    for i = 1:length(F)
        jacs[i] = copy(all_jacobians(F.f[i]))
        for (key, val) in jacs[i]
            jacs[i][key] = F.a[i] * val
        end
    end
    return jacs
end

"""
    FuncLinComb([; jacobians, scalar])

Evaluate the function linear combination. This calls the underlying
`ProgramFunction` objects, so see its documentation.

# Arguments
- `foo`: description.

# Returns
- `bar`: description.
"""
function (FuncLinComb::FunctionLinearCombination)(;
    jacobians::Bool = false,
    scalar::Bool = false,
)::FunctionValueOutputType

    for i = 1:length(FuncLinComb)
        FuncLinComb.f[i](jacobians = jacobians, scalar = scalar)
    end

    return value(FuncLinComb, scalar = scalar)
end

"""
    J([; jacobians, scalar])

Evaluate the cost function. This just passes the call to the underlying
`FunctionLinearCombination`, so see its documentation.
"""
function (J::QuadraticCost)(; jacobians::Bool = false)::FunctionValueOutputType
    terms = core_terms(J)
    return terms(jacobians = jacobians, scalar = true)
end

"""
    add!(J, f[, a])

Add a new term to the quadratic cost. This just calls the underlying `add!`
function for the associated `FunctionLinearCombination`.

# Arguments
- `J`: the quadratic cost.
- `f`: the new term to be added.
- `a`: (optional) the coefficient multiplying this new term

# Returns
- `new_term`: the newly added quadratic cost term.
"""
function add!(J::QuadraticCost, f::ProgramFunction, a::Types.RealTypes = 1.0)::QuadraticCost
    add!(core_terms(J), f, a)
    update_jump_cost!(J)
    new_term = term(J, 0)
    return new_term
end

"""
    update_jump_cost!(J)

Update the underlying JuMP cost.

# Arguments
- `J`: the quadratic cost object.
"""
function update_jump_cost!(J::QuadraticCost)::Nothing
    terms = core_terms(J)
    mdl = jump_model(J)
    J_value = terms(scalar = true)
    set_objective_function(mdl, J_value)
    set_objective_sense(mdl, MOI.MIN_SENSE)
    return nothing
end

""" Get the parent JuMP optimization model. """
jump_model(J::QuadraticCost)::Model = jump_model(J.prog[])

""" Get the actual underlying cost function. """
core_terms(J::QuadraticCost)::FunctionLinearCombination = J.terms

"""
    term(J, i)

Get the i-th term of the cost. If zero or negative, count from the last term.

# Arguments
- `J`: the quadratic cost.
- `i`: the term number, with `0` meaning the last term, `-1` the second last,
  etc..

# Returns
- `Ji`: the individual quadratic cost term.
"""
term(J::QuadraticCost, i::Int)::QuadraticCost = begin
    terms = core_terms(J)
    if i > 0
        f, a = terms.f[i], terms.a[i]
    else
        f, a = terms.f[end-i], terms.a[end-i]
    end
    return QuadraticCost(f, J.prog[], a, noupdate = true)
end

"""
    value(J[; raw])

Get the current objective function value. If the underlying optimization
problem has been solved, return the cost value.

# Arguments
- `J`: the quadratic cost object.

# Keywords
- `raw`: (optional) return the raw cost expression, not evaluated even if the
  optimization has been solved.

# Returns
- `val`: the cost value (a variable expression is not yet solved, or the
  optimal cost value if already solved).
"""
function value(J::QuadraticCost; raw::Bool = false)::FunctionValueOutputType
    val = value(core_terms(J); scalar = true)
    if !raw && termination_status(jump_model(J)) != MOI.OPTIMIZE_NOT_CALLED
        val = value(val)
    end
    return val
end

""" Get the current objective function jacobians. """
jacobian(J::QuadraticCost, i::Int, key::JacobianKeys)::JacobianValueType =
    jacobian(core_terms(J), i, key)
all_jacobians(J::QuadraticCost)::Vector{JacobianDictType} = all_jacobians(core_terms(J))

"""
    feasibility_cost(args...)

A dummy function which represents a feasibility cost.

# Arguments
- `args...`: all arguments are ignored.

# Returns
- `J`: a constant-zero function representing the cost of a feasibility
  optimization problem.
"""
function feasibility_cost(args...)::DifferentiableFunctionOutput
    J = DifferentiableFunctionOutput(0.0)
    return J
end
