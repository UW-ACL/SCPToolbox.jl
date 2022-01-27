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

include("examples/examples.jl")

using Printf
using Test
using .Examples

""" Print a heading for the test set. """
test_heading(algo, description) =
    printstyled(@sprintf("(%s) %s\n", algo, description), color = :blue, bold = true)

# Number of trials if statistics are gathered
const NUM_TRIALS = 3

# @testset "DoubleIntegrator" begin
#     test_heading("LCvx", "Double integrator")
#     Examples.DoubleIntegrator.lcvx()
# end

# @testset "RocketLanding" begin
#     test_heading("LCvx", "Rocket landing")
#     Examples.RocketLanding.lcvx()
# end

@testset "Oscillator" begin
    test_heading("PTR", "Oscillator")
    Examples.Oscillator.ptr()
end

@testset "Quadrotor" begin
    test_heading("SCvx", "Quadrotor")
    Examples.Quadrotor.scvx(NUM_TRIALS)

    test_heading("GuSTO", "Quadrotor")
    Examples.Quadrotor.gusto(NUM_TRIALS)
end

@testset "FreeFlyer" begin
    test_heading("SCvx", "FreeFlyer")
    Examples.FreeFlyer.scvx(NUM_TRIALS)

    test_heading("GuSTO", "FreeFlyer")
    Examples.FreeFlyer.gusto(NUM_TRIALS)
end

@testset "Starship" begin
    test_heading("PTR", "Starship flip")
    Examples.Starship.ptr()

    test_heading("SCvx", "Starship flip")
    Examples.Starship.scvx()
end

@testset "RendezvousPlanar" begin
    test_heading("PTR", "Planar rendezvous")
    Examples.RendezvousPlanar.ptr()
end

@testset "Rendezvous3D" begin
    test_heading("PTR", "Apollo rendezvous")
    homotopy_sweep_steps = 3
    Examples.Rendezvous3D.ptr(NUM_TRIALS, homotopy_sweep_steps)
end
