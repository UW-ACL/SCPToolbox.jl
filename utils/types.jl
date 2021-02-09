# Variable types used in the code

using JuMP

abstract type AbstractTrajectoryProblem end

T_Bool = Bool
T_Int = Int
T_String = String
T_Real = Float64
T_RealOrNothing = Union{T_Real, Nothing}
T_RealVector = Vector{T_Real}
T_RealMatrix = Matrix{T_Real}
T_RealTensor = Array{T_Real, 3}
T_IntRange = UnitRange{T_Int}
T_OptiModel = Model
T_OptiVar = VariableRef
T_OptiVarAffTransf = GenericAffExpr{T_Real,VariableRef}
T_OptiVarVector = Vector{T_OptiVar}
T_OptiVarMatrix = Matrix{T_OptiVar}
T_OptiVarAffTransfVector = Vector{T_OptiVarAffTransf}
T_OptiVarAffTransfMatrix = Matrix{T_OptiVarAffTransf}
T_Constraint = Union{ConstraintRef, Nothing}
T_ConstraintVector = Vector{T_Constraint}
T_ConstraintMatrix = Matrix{T_Constraint}
T_Objective = Union{Nothing,
                    Float64,
                    T_OptiVar,
                    GenericAffExpr{T_Real,T_OptiVar},
                    GenericQuadExpr{T_Real,T_OptiVar}}
