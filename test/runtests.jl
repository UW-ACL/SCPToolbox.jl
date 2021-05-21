#= Trajectory optimization unit tests.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

#nolint: Examples

using Printf
using Test
using Examples

@testset "RocketLanding" begin
    @printf("..:: Lossless convexification rocket landing ::..\n\n")
    Examples.RocketLanding.lcvx()
    @printf("success!\n")
end

# @testset "Oscillator" begin
#     @printf("..:: Oscillator with PTR ::..\n\n")
#     Examples.Oscillator.ptr()
#     @printf("success!\n")
# end

# @testset "Rendezvous3D" begin
#     @printf("..:: 3D Rendezvous with discrete logic ::..\n\n")
#     Examples.Rendezvous3D.ptr()
#     @printf("success!\n")
# end
