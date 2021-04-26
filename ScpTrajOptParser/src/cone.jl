#= Convex cone set type.

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
    include("../../ScpTrajOptUtils/src/ScpTrajOptUtils.jl")
    using .ScpTrajOptUtils
end

using LinearAlgebra
using JuMP

export ConvexCone, add!, fixed_cone, indicator!

ST = ScpTypes
const ConeVariable = Union{ST.Variable, ST.VariableVector}

""" Convex cone constraint.

A conic constraint is of the form `{z : z ∈ K}`, where `K` is a convex cone.

The supported cones are:
- `:zero` for constraints `z==0`.
- `:nonpos` for constraints `z<=0`.
- `:l1` for constraints `z=(t, x), norm(x, 1)<=t`.
- `:soc` for constraints `z=(t, x), norm(x, 2)<=t`.
- `:linf` for constraints `z=(t, x), norm(x, ∞)<=t`.
- `:geom` for constraints `z=(t, x), geomean(x)>=t`.
- `:exp` for constraints `z=(x, y, w), y*exp(x/y)<=w, y>0`.
"""
struct ConvexCone{T<:MOI.AbstractSet}
    z::ConeVariable # The expression to be constrained in the cone
    K::T            # The cone set
    dim::Int        # Cone admbient space dimension
    kind::Symbol    # The kind of cone (:nonpos, :l1, etc.)

    """
        add_conic_constraint!(z, kind[; dual])

    Basic constructor.

    # Arguments
    - `z`: the vector to be constraint to lie within the cone.
    - `kind`: the cone type.

    # Keywords
    - `dual`: (optional) whether to use the dual of the cone instead.

    # Returns
    - `constraint`: the conic constraint.
    """
    function ConvexCone(z::ConeVariable,
                        kind::Symbol;
                        dual::Bool=false)::ConvexCone
        if !(kind in (:zero, :nonpos, :l1, :soc, :linf, :geom, :exp))
            err = SCPError(0, SCP_BAD_ARGUMENT, "ERROR: Unsupported cone.")
            throw(err)
        end

        z = (typeof(z) <: Array) ? z : [z]
        dim = length(z)

        if kind==:zero
            K = dual ? MOI.Reals(dim) : MOI.Zeros(dim)
        elseif kind==:nonpos
            K = MOI.Nonpositives(dim)
        elseif kind==:l1 || (dual && kind==:linf)
            K = MOI.NormOneCone(dim)
        elseif kind==:soc
            K = MOI.SecondOrderCone(dim)
        elseif kind==:linf || (dual && kind==:l1)
            K = MOI.NormInfinityCone(dim)
        elseif kind==:geom
            K = MOI.GeometricMeanCone(dim)
            if dual
                z = [-z[1]; z[2:end]]
            end
        elseif kind==:exp
            if dim!=3
                msg = "ERROR: Exponential cone is in R^3."
                err = SCPError(0, SCP_BAD_ARGUMENT, msg)
                throw(err)
            end
            K = dual ? MOI.DualExponentialCone() : MOI.ExponentialCone()
        end

        constraint = new{typeof(K)}(z, K, dim, kind)

        return constraint
    end # function
end # struct

"""
    add!(pbm, cone)

Add a conic constraint to the optimization problem.

# Arguments
- `pbm`: the optimization problem structure.
- `cone`: the conic constraint structure.

# Returns
- `constraint`: the conic constraint reference.
"""
function add!(pbm::Model, cone::T)::ST.Constraint where {T<:ConvexCone}

    constraint = @constraint(pbm, cone.z in cone.K)

    return constraint
end # function

"""
    add!(pbm, cones)

Add several conic constraints to the optimization problem.

# Arguments
- `pbm`: the optimization problem structure.
- `cones`: an array of conic constraint structures.

# Returns
- `constraints`: the conic constraint references.
"""
function add!(
    pbm::Model, cones::Vector{T})::ST.ConstraintVector where {T<:ConvexCone}

    constraints = ST.ConstraintVector(undef, 0)

    for cone in cones
        push!(constraints, add!(pbm, cone))
    end

    return constraints
end # function

"""
    fixed_cone(cone)

Check if the cone is fixed (i.e. just numerical values, not used inside an
optimization).

# Arguments
- `cone`: the cone object.

# Returns
- `is_fixed`: true if the cone is purely numerical.
"""
function fixed_cone(cone::ConvexCone)::Bool
    is_fixed = typeof(cone.z)<:ST.RealArray
    return is_fixed
end # function

"""
    indicator!(pbm, cone)

Generate a vector which indicates conic constraint satisfaction.

Consider the cone K which defines the constraint x∈K. Let K⊂R^n, an
n-dimensional ambient space. Let q∈R^n be an n-dimensional indicator vector,
such that q<=0 implies x∈K. Furthermore, we formulate q such that if x∈K, then
it is feasible to set q<=0. Hence, effectively, we have a bidirectional
relationship: q<=0 if and only if x∈K.

# Arguments
- `pbm`: the optimization problem structure.
- `cone`: the conic constraint structure.

# Returns
- `q`: the indicator vector.
"""
function indicator!(pbm::Model, cone::ConvexCone)::ST.Variable

    # Parameters
    mode = (fixed_cone(cone)) ? :numerical : :jump
    C = ConvexCone

    # Compute the indicator
    if mode==:numerical
        z = cone.z
        if cone.kind==:zero
            q = abs.(z)
        elseif cone.kind==:nonpos
            q = z
        elseif cone.kind in (:l1, :soc, :linf)
            t = z[1]
            x = z[2:end]
            nrm = Dict(:l1 => 1, :soc => 2, :linf => Inf)
            q = norm(x, nrm[cone.kind])-t
        elseif cone.kind==:geom
            t, x = z[1], z[2:end]
            dim = cone.dim-1
            q = t-exp(1/dim*sum(log.(x)))
        elseif cone.kind==:exp
            x, y, w = z
            q = y*exp(x/y)-w
        end
    else
        z = cone.z
        if cone.kind in (:zero, :nonpos)
            q = @variable(pbm, [1:cone.dim], base_name="q")
            add!(pbm, C(z-q, :nonpos))
            if cone.kind==:zero
                add!(pbm, C(-q-z, :nonpos))
            end
        else
            q = @variable(pbm, base_name="q")
            if cone.kind in (:l1, :soc, :linf)
                t = z[1]
                x = z[2:end]
                add!(pbm, C(vcat(t+q, x), cone.kind))
            elseif cone.kind==:geom
                t, x = z[1], z[2:end]
                add!(pbm, C(vcat(x, t-q), cone.kind))
            elseif cone.kind==:exp
                x, y, w = z
                add!(pbm, C(vcat(x, y, w+q), cone.kind))
            end
            q = [q]
        end
    end

    return q
end # function
