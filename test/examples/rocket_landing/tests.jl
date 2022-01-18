#= Tests for lossless convexification rocket landing.

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

export lcvx

function lcvx()::Nothing

    # Define the vehicle
    rocket = Rocket()

    tol = 1e-3
    tf_min = rocket.m_dry * norm(rocket.v0, 2) / rocket.ρ_max
    tf_max = (rocket.m_wet - rocket.m_dry) / (rocket.α * rocket.ρ_min)

    t_opt, cost_opt =
        golden((tf) -> solve_pdg_fft(rocket, tf).cost, tf_min, tf_max; tol = tol)

    pdg = solve_pdg_fft(rocket, t_opt) # Optimal 3-DoF PDG trajectory

    @test !isinf(cost_opt)

    # Continuous-time simulation
    sim = simulate(rocket, pdg)

    # Make plots
    plot_thrust(rocket, pdg, sim)
    plot_mass(rocket, pdg, sim)
    plot_pointing_angle(rocket, pdg, sim)
    plot_velocity(rocket, pdg, sim)
    plot_position(rocket, pdg, sim)

    return nothing
end
