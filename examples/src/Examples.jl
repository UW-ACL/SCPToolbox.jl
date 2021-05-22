#= Trajectory generation examples.

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

module Examples

module DoubleIntegrator
include("double_integrator/parameters.jl")
include("double_integrator/definition.jl")
include("double_integrator/plots.jl")
include("double_integrator/tests.jl")
end # module

module RocketLanding
include("rocket_landing/parameters.jl")
include("rocket_landing/definition.jl")
include("rocket_landing/plots.jl")
include("rocket_landing/tests.jl")
end # module

module Oscillator
include("oscillator/parameters.jl")
include("oscillator/definition.jl")
include("oscillator/plots.jl")
include("oscillator/tests.jl")
end # module

module RendezvousPlanar
include("rendezvous_planar/parameters.jl")
include("rendezvous_planar/definition.jl")
include("rendezvous_planar/plots.jl")
end # module

module Rendezvous3D
include("rendezvous_3d/parameters.jl")
include("rendezvous_3d/definition.jl")
include("rendezvous_3d/plots.jl")
include("rendezvous_3d/tests.jl")
end #module

end # module
