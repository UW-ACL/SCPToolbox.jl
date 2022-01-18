#= Starship landing flip maneuver data structures and custom methods.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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

include("../../../src/SCPToolbox.jl")

using LinearAlgebra
using .SCPToolbox

# ..:: Data structures ::..

#= Starship vehicle parameters. =#
struct StarshipParameters
    # ..:: Indices ::..
    id_r::IntRange        # Position indices of the state vector
    id_v::IntRange        # Velocity indices of the state vector
    id_θ::Int             # Tilt angle index of the state vector
    id_ω::Int             # Tilt rate index of the state vector
    id_m::Int             # Mass index of the state vector
    id_δd::Int            # Delayed gimbal angle index of the state vector
    id_T::Int             # Thrust index of the input vector
    id_δ::Int             # Gimbal angle index of the input vector
    id_δdot::Int          # Gimbal rate index of the input vector
    id_t1::Int            # First phase duration index of parameter vector
    id_t2::Int            # Second phase duration index of parameter vector
    id_xs::IntRange       # State at phase switch indices of parameter vector
    # ..:: Body axes ::..
    ei::Function          # Lateral body axis in body or world frame
    ej::Function          # Longitudinal body axis in body or world frame
    # ..:: Mechanical parameters ::..
    lcg::RealValue        # [m] CG location (from base)
    lcp::RealValue        # [m] CP location (from base)
    m::RealValue          # [kg] Total mass
    J::RealValue          # [kg*m^2] Moment of inertia about CG
    # ..:: Aerodynamic parameters ::..
    CD::RealValue         # [kg/m] Overall drag coefficient 0.5*ρ*cd*A
    # ..:: Propulsion parameters ::..
    T_min1::RealValue     # [N] Minimum thrust (one engine)
    T_max1::RealValue     # [N] Maximum thrust (one engine)
    T_min3::RealValue     # [N] Minimum thrust (three engines)
    T_max3::RealValue     # [N] Maximum thrust (three engines)
    αe::RealValue         # [s/m] Mass depletion propotionality constant
    δ_max::RealValue      # [rad] Maximum gimbal angle
    δdot_max::RealValue   # [rad/s] Maximum gimbal rate
    rate_delay::RealValue # [s] Delay for approximate rate constraint
end

#= Starship flight environment. =#
struct StarshipEnvironmentParameters
    ex::RealVector # Horizontal "along" axis
    ey::RealVector # Vertical "up" axis
    g::RealVector  # [m/s^2] Gravity vector
end

#= Trajectory parameters. =#
mutable struct StarshipTrajectoryParameters
    r0::RealVector    # [m] Initial position
    v0::RealVector    # [m/s] Initial velocity
    θ0::RealValue     # [rad] Initial tilt angle
    vs::RealVector    # [m/s] Phase switch velocity
    θs::RealValue     # [rad] Phase switch tilt angle
    vf::RealVector    # [m/s] Terminal velocity
    tf_min::RealValue # Minimum flight time
    tf_max::RealValue # Maximum flight time
    γ_gs::RealValue   # [rad] Maximum glideslope (measured from vertical)
    θmax2::RealValue  # [rad] Maximum tilt for terminal descent phase
    τs::RealValue     # Normalized time end of first phase
    hs::RealValue     # [m] Phase switch altitude guess
end

#= Starship trajectory optimization problem parameters all in one. =#
struct StarshipProblem
    vehicle::StarshipParameters        # The ego-vehicle
    env::StarshipEnvironmentParameters # The environment
    traj::StarshipTrajectoryParameters # The trajectory
end

# ..:: Methods ::..

#= Constructor for the Starship landing flip maneuver problem.

Returns:
    mdl: the problem definition object. =#
function StarshipProblem()::StarshipProblem

    # >> Environment <<
    ex = [1.0; 0.0]
    ey = [0.0; 1.0]
    g0 = 9.81 # [m/s^2] Gravitational acceleration
    g = -g0 * ey
    env = StarshipEnvironmentParameters(ex, ey, g)

    # >> Starship <<
    # Indices
    id_r = 1:2
    id_v = 3:4
    id_θ = 5
    id_ω = 6
    id_m = 7
    id_δd = 8
    id_T = 1
    id_δ = 2
    id_δdot = 3
    id_t1 = 1
    id_t2 = 2
    id_xs = (1:id_δd) .+ 2
    # Body axes
    ei = (θ) -> cos(θ) * [1.0; 0.0] + sin(θ) * [0.0; 1.0]
    ej = (θ) -> -sin(θ) * [1.0; 0.0] + cos(θ) * [0.0; 1.0]
    # Mechanical parameters
    rs = 4.5 # [m] Fuselage radius
    ls = 50.0 # [m] Fuselage height
    m = 120e3
    lcg = 0.4 * ls
    lcp = 0.45 * ls
    J = 1 / 12 * m * (6 * rs^2 + ls^2)
    # Aerodynamic parameters
    vterm = 85 # [m/s] Terminal velocity (during freefall)
    CD = m * g0 / vterm^2
    CD *= 1.2 # Fudge factor
    # Propulsion parameters
    Isp = 330 # [s] Specific impulse
    T_min1 = 880e3 # [N] One engine min thrust
    T_max1 = 2210e3 # [N] One engine max thrust
    T_min3 = 3 * T_min1
    T_max3 = 3 * T_max1
    αe = -1 / (Isp * g0)
    δ_max = deg2rad(10.0)
    δdot_max = 2 * δ_max
    rate_delay = 0.05

    starship = StarshipParameters(
        id_r,
        id_v,
        id_θ,
        id_ω,
        id_m,
        id_δd,
        id_T,
        id_δ,
        id_δdot,
        id_t1,
        id_t2,
        id_xs,
        ei,
        ej,
        lcg,
        lcp,
        m,
        J,
        CD,
        T_min1,
        T_max1,
        T_min3,
        T_max3,
        αe,
        δ_max,
        δdot_max,
        rate_delay,
    )

    # >> Trajectory <<
    # Initial values
    r0 = 100.0 * ex + 600.0 * ey
    v0 = -vterm * ey
    θ0 = deg2rad(90.0)
    # Phase switch (guess) values
    θs = deg2rad(-10.0)
    vs = -10.0 * ey
    # Terminal values
    vf = -0.1 * ey
    tf_min = 0.0
    tf_max = 40.0
    γ_gs = deg2rad(27.0)
    θmax2 = deg2rad(15.0)
    τs = 0.5
    hs = 100.0
    traj = StarshipTrajectoryParameters(
        r0,
        v0,
        θ0,
        vs,
        θs,
        vf,
        tf_min,
        tf_max,
        γ_gs,
        θmax2,
        τs,
        hs,
    )

    mdl = StarshipProblem(starship, env, traj)

    return mdl
end
