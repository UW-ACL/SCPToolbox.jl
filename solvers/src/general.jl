#= General code used by the sequential convex programming solvers.

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

LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("../../utils/src/Utils.jl")
    include("../../parser/src/Parser.jl")

    using .Utils
    using .Parser

    import .Utils.Types: sample

    import .Parser: DiscretizationType, FOH, IMPULSE
    import .Parser.ConicLinearProgram: ConicProgram
    import .Parser.ConicLinearProgram: QuadraticCost
    import .Parser.ConicLinearProgram: VariableArgumentBlock
    import .Parser.ConicLinearProgram: ConstantArgumentBlock
    import .Parser.ConicLinearProgram: @add_constraint, ZERO
end

using LinearAlgebra

using Utils
using Parser

# ..:: Globals  ::..

const ST = Types

const IntRange = ST.IntRange

const RealTypes = ST.RealTypes
const RealVector = ST.RealVector
const RealMatrix = ST.RealMatrix
const RealTensor = ST.RealTensor

const VarArgBlk = VariableArgumentBlock
const CstArgBlk = ConstantArgumentBlock

const Objective = Union{ST.Objective, QuadraticCost}

const Trajectory = ST.ContinuousTimeTrajectory

abstract type AbstractSCPProblem end
abstract type SCPSubproblemSolution end

# ..:: Methods  ::..

"""
    get_time()

The the current time in nanoseconds.
"""
function get_time()::Int
    return Int(time_ns())
end # function
