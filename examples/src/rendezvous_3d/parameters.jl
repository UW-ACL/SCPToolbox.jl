#= Spacecraft rendezvous data structures.

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
    include("../../../utils/src/Utils.jl")

    using .Utils
end

using LinearAlgebra

using Utils

# ..:: Globals ::..

const Ty = Types
const IntRange = Ty.IntRange
const RealValue = Ty.RealTypes
const RealVector = Ty.RealVector
const RealMatrix = Ty.RealMatrix
const Quaternion = Ty.Quaternion

# ..:: Data structures ::..

""" Chaser vehicle parameters. """
struct ChaserParameters
    # >> Indices <<
    id_r::IntRange   # Inertial position (state)
    id_v::IntRange   # Inertial velocity (state)
    id_q::IntRange   # Quaternion (state)
    id_ω::IntRange   # Body frame angular velocity (state)
    id_T::IntRange   # Total thrust (input)
    id_M::IntRange   # Total torque (input)
    id_t::Int        # Time dilation (parameter)
    # >> Mechanical parameters <<
    m::RealValue     # [kg] Mass
    J::RealMatrix    # [kg*m^2] Principle moments of inertia matrix
    # >> Control parameters <<
    T_max::RealValue # [N] Maximum thrust
    M_max::RealValue # [N*m] Maximum torque
end # struct

""" Planar rendezvous flight environment. """
struct RendezvousEnvironmentParameters
    xi::RealVector   # Inertial axis "normal out of dock port"
    yi::RealVector   # Inertial axis "dock port left (when facing)"
    zi::RealVector   # Inertial axis "dock port down (when facing)"
    n::RealValue     # [rad/s] Orbital mean motion
end

""" Parameters of the chaser trajectory. """
struct RendezvousTrajectoryParameters
    # >> Boundary conditions <<
    r0::RealVector      # Initial position
    rf::RealVector      # Terminal position
    v0::RealVector      # Initial velocity
    vf::RealVector      # Terminal velocity
    q0::Quaternion      # Initial attitude
    qf::Quaternion      # Terminal attitude
    ω0::RealVector      # Initial angular velocity
    ωf::RealVector      # Terminal angular velocity
    # >> Time of flight <<
    tf_min::RealValue   # Minimum flight time
    tf_max::RealValue   # Maximum flight time
    # >> Homotopy <<
    # TODO
end # struct

""" Rendezvous trajectory optimization problem parameters all in one. """
struct RendezvousProblem
    vehicle::ChaserParameters            # The ego-vehicle
    env::RendezvousEnvironmentParameters # The environment
    traj::RendezvousTrajectoryParameters # The trajectory
end

# ..:: Methods ::..

"""
    RendezvousProblem()

Constructor for the "full" 3D rendezvous problem.

# Returns
- `mdl`: the problem definition object.
"""
function RendezvousProblem()::RendezvousProblem

    # ..:: Environment ::..
    xi = [1.0; 0.0; 0.0]
    yi = [0.0; 1.0; 0.0]
    zi = [0.0; 0.0; 1.0]
    μ = 3.986e14 # [m³/s²] Standard gravitational parameter
    Re = 6378e3 # [m] Earth radius
    R = Re+400e3 # [m] Orbit radius
    n = sqrt(μ/R^3)
    env = RendezvousEnvironmentParameters(xi, yi, zi, n)

    # ..:: Chaser spacecraft ::..
    # >> Indices <<
    id_r = 1:3
    id_v = 4:6
    id_q = 7:10
    id_ω = 11:13
    id_T = 1:3
    id_M = 4:6
    id_t = 1
    # >> Mechanical parameters <<
    m = 30e3
    J = diagm([5e4; 1e5; 1e5])
    # >> Control parameters <<
    T_max = 500.0
    M_max = 1500.0

    sc = ChaserParameters(id_r, id_v, id_q, id_ω, id_T, id_M, id_t,
                          m, J, T_max, M_max)

    # ..:: Trajectory ::..
    # >> Boundary conditions <<
    r0 = 10.0*xi
    rf = 0.0*xi
    v0 = 0.0*xi
    vf = -0.1*xi
    ω0 = zeros(3)
    ωf = zeros(3)
    # Docking port (inertial) frame
    # Baseline docked configuration
    q_dock = Quaternion(deg2rad(180), yi)*Quaternion(deg2rad(180), xi)
    q_init = q_dock*Quaternion(deg2rad(180), yi)
    q0 = q_init
    qf = q_dock
    # >> Time of flight <<
    tf_min = 100.0
    tf_max = 500.0

    traj = RendezvousTrajectoryParameters(
        r0, rf, v0, vf, q0, qf, ω0, ωf,
        tf_min, tf_max)

    mdl = RendezvousProblem(sc, env, traj)

    return mdl
end # function
