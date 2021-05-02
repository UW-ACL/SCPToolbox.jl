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

# User-facing problem definition
# include("problem.jl")

# General optimization problem building
module ConicLinearProgram
include("cone.jl")
include("function.jl")
include("program.jl")
include("printing.jl")
end # module

using .ConicLinearProgram
export ConicProgram, blocks, variable!, value
export @new_variable, @new_parameter

end # module
