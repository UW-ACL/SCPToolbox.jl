#= Optimization problem parser.

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

module Parser

if isdefined(@__MODULE__, :LanguageServer)
    include("../../utils/src/Utils.jl")
end

export ConicLinearProgram

# User-facing problem definition
include("problem.jl")

# General optimization problem building
module ConicLinearProgram
include("general.jl")
include("scaling.jl")
include("perturbation.jl")
include("block.jl")
include("argument.jl")
include("function.jl")
include("cone.jl")
include("constraint.jl")
include("cost.jl")
include("program.jl")
include("variation.jl")
include("printing.jl")
end # module

using .ConicLinearProgram

export ConicProgram, numel, constraints, cost, solve!, jump_model
export ConvexCone, add!, isfixed, isfree, dual, indicator!
export jacobian, set_jacobian!
export SupportedCone, UNCONSTRAINED, ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
export SupportedDualCone, UNCONSTRAINED_DUAL, ZERO_DUAL, NONPOS_DUAL, L1_DUAL,
    SOC_DUAL, LINF_DUAL, GEOM_DUAL, EXP_DUAL
export ArgumentBlock, VariableArgumentBlock, ConstantArgumentBlock
export QuadraticCost
export variation

export @new_variable, @new_parameter, @add_constraint,
    @add_cost, @set_feasibility, @scale, @perturb_free,
    @perturb_fix, @perturb_relative, @perturb_absolute,
    @value, @jacobian

end # module
