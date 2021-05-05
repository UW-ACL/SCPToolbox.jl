#= Basic and composite types used by the code.

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
    include("globals.jl")
end

using LinearAlgebra
using JuMP

import ..SCPStatus

""" Get the types composing a Union. """
Base.collect(t::Union{Type, DataType, Union{}}) = _collect(t, [])
_collect(t::Type, list) = t<:Union{} ? push!(list, t) :
    _collect(t.b, push!(list, t.a))
_collect(t::Union{DataType,Core.TypeofBottom}, list) = push!(list, t)

const Optional{T} = Union{T, Nothing}

const RealTypes = Union{Int, Float64}

const IntVector = Vector{Int}
const IntRange = UnitRange{Int}
const Index = Union{Int, IntRange, Colon}

const RealArray{n} = Union{[Array{T, n} for T in collect(RealTypes)]...}
const RealVector = RealArray{1}
const RealMatrix = RealArray{2}
const RealTensor = RealArray{3}

const VRef = VariableRef
const AExpr = GenericAffExpr{Float64, VRef}
const QExpr = GenericQuadExpr{Float64, VRef}
const Variable = Union{RealTypes, VariableRef, AExpr, QExpr}
const VariableArray{n} = Union{[Array{T, n} for T in collect(Variable)]...}
const VariableVector = VariableArray{1}
const VariableMatrix = VariableArray{2}

const Constraint = ConstraintRef
const ConstraintArray{n} = Array{Constraint, n}
const ConstraintVector = ConstraintArray{1}
const ConstraintMatrix = ConstraintArray{2}

const Objective = Union{Missing, Variable}

const ExitStatus = Union{SCPStatus, MOI.TerminationStatusCode}

const Func = Union{Nothing, Function}

const SpecialIntegrationActions = Vector{Tuple{Index, Function}}

"""
    RealArray(x)

Real array constructor.

# Arguments
- `X`: initialization arguments that you would normally pass to a constructor
  like `Vector{Float64}`.

# Returns
The real array.
"""
function (::Type{RealArray})(
    X::Array{T, n})::Array{T, n} where {T<:RealTypes, n}

    return Array{T, n}(X)
end # function

"""
#     VariableArray(x)

Variable array constructor.

# Arguments
- `X`: initialization arguments that you would normally pass to a constructor
  like `Vector{Float64}`.

# Returns
The variable array.
"""
function (::Type{VariableArray})(
    X::Array{T, n})::Array{T, n} where {T<:Variable, n}

    return Array{T, n}(X)
end # function
