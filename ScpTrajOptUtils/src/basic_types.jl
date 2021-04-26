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

const RealTypes = [Int, Float64]

const BoolVector = Vector{Bool}
const IntVector = Vector{Int}
const IntRange = UnitRange{Int}

const RealValue = Union{[T for T in RealTypes]...}
const RealArray{n} = Union{[Array{T, n} for T in RealTypes]...}
const RealVector = RealArray{1}
const RealMatrix = RealArray{2}
const RealTensor = RealArray{3}

const AffineExpression = GenericAffExpr{Float64, VariableRef}
const VariableTypes = [RealTypes..., VariableRef, AffineExpression]
const Variable = Union{[T for T in VariableTypes]...}
const VariableArray{n} = Union{[Array{T, n} for T in VariableTypes]...}
const VariableVector = VariableArray{1}
const VariableMatrix = VariableArray{2}

const Constraint = ConstraintRef
const ConstraintArray{n} = Array{Constraint, n}
const ConstraintVector = ConstraintArray{1}
const ConstraintMatrix = ConstraintArray{2}

const Objective = Union{Missing, RealTypes..., VariableRef,
                        GenericAffExpr{Float64, VariableRef},
                        GenericQuadExpr{Float64, VariableRef}}

const ExitStatus = Union{SCPStatus, MOI.TerminationStatusCode}

const Func = Union{Nothing, Function}
const FuncVector = Vector{Function}

const ElementIndex = Union{Int, IntRange, Colon}
const SpecialIntegrationActions = Vector{Tuple{ElementIndex, Function}}
