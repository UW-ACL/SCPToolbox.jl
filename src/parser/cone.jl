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

import JuMP: dual

export ConvexCone, add!, isfixed, isfree, dual, indicator!
export SupportedCone, UNCONSTRAINED, ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
export SupportedDualCone, UNCONSTRAINED_DUAL, ZERO_DUAL, NONPOS_DUAL, L1_DUAL,
    SOC_DUAL, LINF_DUAL, GEOM_DUAL, EXP_DUAL

const ConeVariable = Union{Types.Variable, Types.VariableVector}

"""
A conic constraint is of the form `{z : z ∈ K}`, where `K` is a convex cone.

The supported cones are:
- `UNCONSTRAINED` for unconstrained.
- `ZERO` for constraints `z==0`.
- `NONPOS` for constraints `z<=0`.
- `L1` for constraints `z=(t, x), norm(x, 1)<=t`.
- `SOC` for constraints `z=(t, x), norm(x, 2)<=t`.
- `LINF` for constraints `z=(t, x), norm(x, ∞)<=t`.
- `GEOM` for constraints `z=(t, x), geomean(x)>=t`.
- `EXP` for constraints `z=(x, y, w), y*exp(x/y)<=w, y>0`.
"""
@enum(SupportedCone, UNCONSTRAINED, ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP)

"""
Similar to `SupportedCone`, except with a `_DUAL` qualiifier postfix.
"""
@enum(SupportedDualCone, UNCONSTRAINED_DUAL, ZERO_DUAL, NONPOS_DUAL, L1_DUAL,
      SOC_DUAL, LINF_DUAL, GEOM_DUAL, EXP_DUAL)

# Maps from the original cone to the dual cone
const DUAL_CONE_MAP = Dict(
    # Maps from "CONE" to "CONE_DUAL"
    # These are **only used internally**
    UNCONSTRAINED => UNCONSTRAINED_DUAL,
    ZERO => ZERO_DUAL,
    NONPOS => NONPOS_DUAL,
    L1 => L1_DUAL,
    SOC => SOC_DUAL,
    LINF => LINF_DUAL,
    GEOM => GEOM_DUAL,
    EXP => EXP_DUAL,
    # Maps from "CONE_DUAL" to the actual dual cone
    UNCONSTRAINED_DUAL => ZERO,
    ZERO_DUAL => UNCONSTRAINED,
    NONPOS_DUAL => NONPOS,
    L1_DUAL => LINF,
    SOC_DUAL => SOC,
    LINF_DUAL => L1,
    GEOM_DUAL => GEOM,
    EXP_DUAL => EXP)

const CONE_NAMES = Dict(UNCONSTRAINED => "unconstrained",
                        ZERO => "zero",
                        NONPOS => "nonpositive orthant",
                        L1 => "one-norm",
                        SOC => "second-order",
                        LINF => "inf-norm",
                        GEOM => "geometric",
                        EXP => "exponential")

"""
`ConvexCone` stores the information necessary to form a convex cone constraint
in JUMP.
"""
struct ConvexCone{T<:MOI.AbstractSet}
    z::ConeVariable     # The expression to be constrained in the cone
    K::T                # The cone set
    dim::Int            # Cone admbient space dimension
    kind::SupportedCone # The kind of cone (NONPOS, L1, etc.)

    """
        add_conic_constraint!(z, kind)

    Basic constructor.

    # Arguments
    - `z`: the vector to be constraint to lie within the cone.
    - `kind`: the cone type (or its dual).

    # Returns
    - `constraint`: the conic constraint.
    """
    function ConvexCone(z::ConeVariable,
                        kind::Union{SupportedCone,
                                    SupportedDualCone})::ConvexCone

        z = (typeof(z) <: Array) ? z : [z]
        dim = length(z)

        if kind==EXP && dim!=3
            msg = "Exponential cone is in R^3"
            err = SCPError(0, SCP_BAD_ARGUMENT, msg)
            throw(err)
        end

        # Convert to dual cone
        if kind isa SupportedDualCone
            kind_dual = DUAL_CONE_MAP[kind]
            if kind==GEOM_DUAL
                t, x = z[1], z[2:end]
                n = length(x)
                z = [-t/n; x]
            elseif kind==EXP_DUAL
                u, v, w = z
                z = [-u; -v; exp(1)*w]
            end
            return ConvexCone(z, kind_dual)
        end

        if kind==UNCONSTRAINED
            K = MOI.Reals(dim)
        elseif kind==ZERO
            K = MOI.Zeros(dim)
        elseif kind==NONPOS
            K = MOI.Nonpositives(dim)
        elseif kind==L1
            K = MOI.NormOneCone(dim)
        elseif kind==SOC
            K = MOI.SecondOrderCone(dim)
        elseif kind==LINF
            K = MOI.NormInfinityCone(dim)
        elseif kind==GEOM
            K = MOI.GeometricMeanCone(dim)
        elseif kind==EXP
            K = MOI.ExponentialCone()
        end

        constraint = new{typeof(K)}(z, K, dim, kind)

        return constraint
    end
end # struct

""" Get the kind of cone """
kind(cone::ConvexCone)::SupportedCone = cone.kind

""" Ambient space dimension of the cone """
Base.ndims(cone::ConvexCone)::Int = cone.dim

"""
    add!(pbm, cone)

Add a conic constraint to the optimization problem.

# Arguments
- `pbm`: the optimization problem structure.
- `cone`: the conic constraint structure.

# Returns
- `constraint`: the conic constraint reference.
"""
function add!(pbm::Model, cone::ConvexCone)::Types.Constraint

    if isfree(cone)
        # Just skip the "all of R^n" cone, since it means unconstrained
        constraint = nothing
    else
        constraint = @constraint(pbm, cone.z in cone.K)
    end

    return constraint
end

"""
    add!(pbm, cones)

Add several conic constraints to the optimization problem.

# Arguments
- `pbm`: the optimization problem structure.
- `cones`: an array of conic constraint structures.

# Returns
- `constraints`: the conic constraint references.
"""
function add!(pbm::Model,
              cones::Vector{T})::Types.ConstraintVector where {
                  T<:ConvexCone}

    constraints = Types.ConstraintVector(undef, 0)

    for cone in cones
        push!(constraints, add!(pbm, cone))
    end

    return constraints
end

"""
    isfixed(cone)

Check if the cone is fixed (i.e. just numerical values, not used inside an
optimization).

# Arguments
- `cone`: the cone object.

# Returns
- `is_fixed`: true if the cone is purely numerical.
"""
function isfixed(cone::ConvexCone)::Bool
    is_fixed = typeof(cone.z)<:Types.RealArray
    return is_fixed
end

""" Find out whether the cone is all of ``\\reals^n``. """
isfree(cone::ConvexCone)::Bool = kind(cone)==UNCONSTRAINED

""" Convert from cone to its dual cone. """
dual(cone::SupportedCone)::SupportedDualCone = DUAL_CONE_MAP[cone]

"""
    indicator!(pbm, cone)

TODO update to latest ConvexCone structure (there is now also a UNCONSTRAINED cone)

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
function indicator!(pbm::Model, cone::ConvexCone)::Types.Variable

    # Parameters
    mode = (isfixed(cone)) ? :numerical : :jump

    # Compute the indicator
    if mode==:numerical
        z = cone.z
        if cone.kind==ZERO
            q = abs.(z)
        elseif cone.kind==NONPOS
            q = z
        elseif cone.kind in (L1, SOC, LINF)
            t = z[1]
            x = z[2:end]
            nrm = Dict(L1 => 1, SOC => 2, LINF => Inf)
            q = norm(x, nrm[cone.kind])-t
        elseif cone.kind==GEOM
            t, x = z[1], z[2:end]
            dim = cone.dim-1
            q = t-exp(1/dim*sum(log.(x)))
        elseif cone.kind==EXP
            x, y, w = z
            q = y*exp(x/y)-w
        end
    else
        z = cone.z
        if cone.kind in (ZERO, NONPOS)
            q = @variable(pbm, [1:cone.dim], base_name="q")
            add!(pbm, ConvexCone(z-q, NONPOS))
            if cone.kind==ZERO
                add!(pbm, ConvexCone(-q-z, NONPOS))
            end
        else
            q = @variable(pbm, base_name="q")
            if cone.kind in (L1, SOC, LINF)
                t = z[1]
                x = z[2:end]
                add!(pbm, ConvexCone(vcat(t+q, x), cone.kind))
            elseif cone.kind==GEOM
                t, x = z[1], z[2:end]
                add!(pbm, ConvexCone(vcat(x, t-q), cone.kind))
            elseif cone.kind==EXP
                x, y, w = z
                add!(pbm, ConvexCone(vcat(x, y, w+q), cone.kind))
            end
            q = [q]
        end
    end

    return q
end
