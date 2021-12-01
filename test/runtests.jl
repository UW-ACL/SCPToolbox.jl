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

using Printf
using Test
using Examples

""" Print a heading for the test set. """
test_heading(algo, description) = printstyled(
    @sprintf("(%s) %s\n", algo, description),
    color=:blue, bold=true)

@testset "DoubleIntegrator" begin
    test_heading("LCvx", "Double integrator")
    Examples.DoubleIntegrator.lcvx()
end

@testset "RocketLanding" begin
    test_heading("LCvx", "Rocket landing")
    Examples.RocketLanding.lcvx()
end

@testset "Oscillator" begin
    test_heading("PTR", "Oscillator")
    Examples.Oscillator.ptr()
end

@testset "Rendezvous3D" begin
    test_heading("PTR", "Apollo rendezvous")
    Examples.Rendezvous3D.ptr()
end

@testset "Starship" begin
    test_heading("PTR", "Starship flip")
    Examples.Starship.ptr()
end
