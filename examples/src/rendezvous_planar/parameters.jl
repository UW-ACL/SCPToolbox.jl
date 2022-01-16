#= Planar spacecraft rendezvous data structures and custom methods.

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

using Utils

# ..:: Globals ::..

const T = Types
const RealValue = T.RealTypes
const RealVector = T.RealVector
const RealMatrix = T.RealMatrix

# ..:: Data structures ::..

""" Planar rendezvous parameters. """
struct PlanarRendezvousParameters
    # ..:: Indices ::..
    id_r::T.IntRange     # Position (state)
    id_v::T.IntRange     # Velocity (state)
    id_θ::Int            # Rotation angle (state)
    id_ω::Int            # Rotation rate (state)
    id_f::T.IntRange     # Thrust forces for RCS pods (input)
    id_fr::T.IntRange    # Reference thrust forces for RCS pods (input)
    id_l1f::T.IntRange   # Thrust force absolute values for RCS pods (input)
    id_l1feq::T.IntRange # Thrust force difference one-norm (input)
    id_t::Int            # Time dilation (parameter)
    # ..:: Mechanical parameters ::..
    m::RealValue         # [kg] Mass
    J::RealValue         # [kg*m²] Moment of inertia about CoM
    lu::RealValue        # [m] CoM longitudinal distance aft of thrusters
    lv::RealValue        # [m] CoM transverse distance from thrusters
    uh::Function         # Longitudinal "forward" axis in the inertial frame
    vh::Function         # Transverse "up" axis in the inertial frame
    # ..:: Control parameters ::..
    f_max::RealValue     # [N] Maximum thrust force
    f_db::RealValue      # [N] Deadband thrust force
end

""" Planar rendezvous flight environment. """
struct PlanarRendezvousEnvironmentParameters
    xh::RealVector       # Inertial horizontal axis
    yh::RealVector       # Inertial vertical axis
    n::RealValue         # [rad/s] Orbital mean motion
end

""" Trajectory parameters. """
mutable struct PlanarRendezvousTrajectoryParameters
    r0::RealVector       # [m] Initial position
    v0::RealVector       # [m/s] Initial velocity
    θ0::RealValue        # [rad] Initial rotation angle
    ω0::RealValue        # [rad/s] Initial rotation rate
    vf::RealValue        # [m/s] Final approach speed
    tf_min::RealValue    # [s] Minimum flight time
    tf_max::RealValue    # [s] Maximum flight time
    κ::RealValue         # Sigmoid homotopy parameter
    γ::RealValue         # Control weight for deadband relaxation
end

""" Planar rendezvous trajectory optimization problem parameters all in
one. """
struct PlanarRendezvousProblem
    vehicle::PlanarRendezvousParameters        # The ego-vehicle
    env::PlanarRendezvousEnvironmentParameters # The environment
    traj::PlanarRendezvousTrajectoryParameters # The trajectory
end

# ..:: Methods ::..

"""
    PlanarRendezvousProblem()

Constructor for the planar rendezvous problem.

# Returns
- `mdl`: the problem definition object.
"""
function PlanarRendezvousProblem()::PlanarRendezvousProblem

    # ..:: Environment ::..
    xh = [1.0; 0.0]
    yh = [0.0; 1.0]
    μ = 3.986e14 # [m³/s²] Standard gravitational parameter
    Re = 6378e3 # [m] Earth radius
    R = Re+400e3 # [m] Orbit radius
    n = sqrt(μ/R^3)
    env = PlanarRendezvousEnvironmentParameters(xh, yh, n)

    # ..:: Spacecraft vehicle ::..
    # >> Indices <<
    id_r = 1:2
    id_v = 3:4
    id_θ = 5
    id_ω = 6
    id_f = 1:3
    id_fr = 4:6
    id_l1f = 7:9
    id_l1feq = 10:12
    id_t = 1
    # >> Mechanical parameters <<
    m = 30e3
    J = 1e5
    lu = 0.6
    lv = 2.1
    uh = (θ) -> -cos(θ)*xh+sin(θ)*yh
    vh = (θ) -> -sin(θ)*xh-cos(θ)*yh
    # >> Control parameters <<
    f_max = 750.0 # 445.0
    f_db = 200.0

    sc = PlanarRendezvousParameters(
        id_r, id_v, id_θ, id_ω, id_f, id_fr, id_l1f, id_l1feq, id_t,
        m, J, lu, lv, uh, vh, f_max, f_db)

    # ..:: Trajectory ::..
    r0 = 100.0*xh+10.0*yh
    v0 = 0.0*xh
    θ0 = deg2rad(180.0)
    ω0 = 0.0
    vf = 0.1
    tf_min = 100.0
    tf_max = 500.0
    κ = NaN
    γ = 3e-1
    traj = PlanarRendezvousTrajectoryParameters(
        r0, v0, θ0, ω0, vf, tf_min, tf_max, κ, γ)

    mdl = PlanarRendezvousProblem(sc, env, traj)

    return mdl
end
