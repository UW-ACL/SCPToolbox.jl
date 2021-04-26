#= Data structures and algorithm to work with an optimization problem.

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

module OptimizationProblem

include("types.jl")

using JuMP
using ECOS
using Printf
using .Types

import Base: show

export T_AtomicVariable, T_DecisionVector, T_OptimizationProblem
export add_variable!

abstract type T_AbstractDecisionVector end
abstract type T_AbstractOptimizationProblem end

""" An atomic (vector) variable in the optimization problem. """
struct T_AtomicVariable{T<:T_AbstractDecisionVector}
    name::T_String # Variable "name" (short description)
    var::T_OptiVar # The actual variable structure (JuMP)
    id::T_IntRange # Index range of variable in the parent decision vector
    parent::T      # Parent decision vector

    """
        T_AtomicVariable(parent[, shape][; name])

    Basic constructor.

    # Arguments
    - `parent`: the parent decision vector.
    - `shape`: (optional) the shape of the variable. If not passed in, assumed
      to be a scalar. Can also provide one value (vector dimension) or two
      values (matrix dimension).

    # Keywords
    - `name`: (optional) a name for the variable.

    # Returns
    - `x`: the new variable.
    """
    function T_AtomicVariable(
        parent::T,
        shape::Union{T_Int, Tuple{T_Int, T_Int}}=1;
        name::T_String="")::T_AtomicVariable where {
            T<:T_AbstractDecisionVector}

        # Make the underlying JuMP variable
        if typeof(shape)<:Tuple
            var = @variable(parent.problem.mdl, [1:shape[1], 1:shape[2]],
                            base_name=name)
        else
            var = @variable(parent.problem.mdl, [1:shape], base_name=name)
        end

        # Determine the indices in the parent decision vector
        i0 = parent.n
        m = length(var)
        id = (1:m).+i0

        x = new{typeof(parent)}(name, var, id, parent)

        # Update parent decision vector
        push!(parent.vars, x)
        parent.n += m

        return x
    end
end

""" The overall decision vector for the optimization problem. """
mutable struct T_DecisionVector{T<:T_AbstractOptimizationProblem} <: T_AbstractDecisionVector
    vars::Vector{T_AtomicVariable} # Variables composing the decision vector
    n::T_Int                       # Dimension
    problem::Union{T, Nothing}     # The associated optimization problem

    """
        T_DecisionVector()

    Empty constructor.

    # Arguments
    - `problem`: the underlying JuMP optimization problem.

    # Returns
    - `x`: an empty decision vector.
    """
    function T_DecisionVector{T}()::T_DecisionVector where {
        T<:T_AbstractOptimizationProblem}

        vars = Vector{T_AtomicVariable}(undef, 0)
        x = new{T}(vars, 0, nothing)

        return x
    end
end

""" The optimization problem data structure. """
struct T_OptimizationProblem <: T_AbstractOptimizationProblem
    mdl::Model          # The underlying JuMP optimization model structure
    x::T_DecisionVector # The decision variable vector

    """
        T_OptimizationProblem()

    Create a new empty problem.

    # Returns
    - `pbm`: the problem object.
    """
    function T_OptimizationProblem()::T_OptimizationProblem

        # Make problem data
        mdl = Model(ECOS.Optimizer)
        x = T_DecisionVector{T_OptimizationProblem}()

        # Initialize the problem
        pbm = new(mdl, x)

        # Associate data back to the problem
        x.problem = pbm

        return pbm
    end
end

"""
    add_variable!(pbm[, shape][; name])

Add a new variable to the optimization problem.

# Arguments
- `pbm`: the optimization problem structure.
- `shape`: (optional) the variable shape (possibilities are scalar, vector,
  matrix).

# Keywords
- `name`: (optional) variable name.

# Returns
- `new_var`: the new variable.
"""
function add_variable!(pbm::T_OptimizationProblem,
                       shape::Union{T_Int, Tuple{T_Int, T_Int}}=1;
                       name::T_String="")::T_AtomicVariable

    new_var = T_AtomicVariable(pbm.x, shape; name=name)

    return new_var
end

"""
    show(io, x)

Pretty print an atomic variable.

# Arguments
- `x`: atomic variable.
"""
function show(io::IO, x::T_AtomicVariable)::Nothing
    compact = get(io, :compact, false)

    @printf(io, "Atomic variable:\n")
    @printf(io, "  name = %s\n", x.name)
    @printf(io, "  id = %d:%d", x.id[1], x.id[end])
end

end
