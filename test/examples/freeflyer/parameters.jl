"""
6-Degree of Freedom free-flyer data structures and custom methods.

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
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

include("../../../src/SCPToolbox.jl")

using LinearAlgebra
using .SCPToolbox

# ..:: Data structures ::..

""" Free-flyer vehicle parameters. """
struct FreeFlyerParameters
    id_r::IntRange # Position indices of the state vector
    id_v::IntRange # Velocity indices of the state vector
    id_q::IntRange # Quaternion indices of the state vector
    id_ω::IntRange # Angular velocity indices of the state vector
    id_T::IntRange # Thrust indices of the input vector
    id_M::IntRange # Torque indicates of the input vector
    id_t::Int      # Time dilation index of the parameter vector
    id_δ::IntRange # Room SDF indices of the parameter vector
    v_max::Real    # [m/s] Maximum velocity
    ω_max::Real    # [rad/s] Maximum angular velocity
    T_max::Real    # [N] Maximum thrust
    M_max::Real    # [N*m] Maximum torque
    m::Real        # [kg] Mass
    J::RealMatrix  # [kg*m^2] Principle moments of inertia matrix
end

""" Space station flight environment. """
struct FreeFlyerEnvironmentParameters
    obs::Vector{Ellipsoid}      # Obstacles (ellipsoids)
    iss::Vector{Hyperrectangle} # Space station flight space
    n_obs::Int                  # Number of obstacles
    n_iss::Int                  # Number of space station rooms
end

""" Trajectory parameters. """
mutable struct FreeFlyerTrajectoryParameters
    r0::RealVector # Initial position
    rf::RealVector # Terminal position
    v0::RealVector # Initial velocity
    vf::RealVector # Terminal velocity
    q0::Quaternion # Initial attitude
    qf::Quaternion # Terminal attitude
    ω0::RealVector # Initial angular velocity
    ωf::RealVector # Terminal angular velocity
    tf_min::Real   # Minimum flight time
    tf_max::Real   # Maximum flight time
    γ::Real        # Tradeoff weight terminal vs. running cost
    hom::Real      # Homotopy parameter for signed-distance function
    ε_sdf::Real    # Tiny weight to tighten the room SDF lower bounds
end

""" Free-flyer trajectory optimization problem parameters all in one. """
struct FreeFlyerProblem
    vehicle::FreeFlyerParameters        # The ego-vehicle
    env::FreeFlyerEnvironmentParameters # The environment
    traj::FreeFlyerTrajectoryParameters # The trajectory
end

# ..:: Methods ::..

""" Constructor for the environment.

# Arguments
    iss: the space station flight corridors, defined as hyperrectangles.
    obs: array of obstacles (ellipsoids).

# Returns
    env: the environment struct.
"""
function FreeFlyerEnvironmentParameters(
        iss::Vector{Hyperrectangle},
        obs::Vector{Ellipsoid}
)::FreeFlyerEnvironmentParameters

    # Derived values
    n_iss = length(iss)
    n_obs = length(obs)

    env = FreeFlyerEnvironmentParameters(obs, iss, n_obs, n_iss)

    return env
end


""" Constructor for the 6-DoF free-flyer problem.

# Returns
    mdl: the free-flyer problem.
"""
function FreeFlyerProblem(
        N::Int
)::FreeFlyerProblem

    # >> Environment <<
    obs_shape = diagm([1.0; 1.0; 1.0]/0.3)
    z_iss = 4.75
    obs = [Ellipsoid(copy(obs_shape), [8.5; -0.15; 5.0]),
           Ellipsoid(copy(obs_shape), [11.2; 1.84; 5.0]),
           Ellipsoid(copy(obs_shape), [11.3; 3.8;  4.8])]
    iss_rooms = [
        Hyperrectangle([6.0; 0.0; z_iss],
                        1.0, 1.0, 1.5;
                        pitch=90.0),
        Hyperrectangle([7.5; 0.0; z_iss],
                        2.0, 2.0, 4.0;
                        pitch=90.0),
        Hyperrectangle([11.5; 0.0; z_iss],
                        1.25, 1.25, 0.5;
                        pitch=90.0),
        Hyperrectangle([10.75; -1.0; z_iss],
                        1.5, 1.5, 1.5;
                        yaw=-90.0, pitch=90.0),
        Hyperrectangle([10.75; 1.0; z_iss],
                        1.5, 1.5, 1.5;
                        yaw=90.0, pitch=90.0),
        Hyperrectangle([10.75; 2.5; z_iss],
                        2.5, 2.5, 4.5;
                        yaw=90.0, pitch=90.0)]
    env = FreeFlyerEnvironmentParameters(iss_rooms, obs)

    # >> Free-flyer <<
    id_r = 1:3
    id_v = 4:6
    id_q = 7:10
    id_ω = 11:13
    id_T = 1:3
    id_M = 4:6
    id_t = 1
    id_δ = (1:(N*env.n_iss)).+1
    v_max = 0.4
    ω_max = deg2rad(1)
    T_max = 20e-3
    M_max = 1e-4
    mass = 7.2
    J = diagm([0.1083, 0.1083, 0.1083])
    fflyer = FreeFlyerParameters(
        id_r, id_v, id_q, id_ω, id_T, id_M, id_t,
        id_δ, v_max, ω_max, T_max, M_max, mass, J)

    # >> Trajectory <<
    r0 = [6.5; -0.2; 5.0]
    v0 = [0.035; 0.035; 0.0]
    q0 = Quaternion(deg2rad(-40), [0.0; 1.0; 1.0])
    ω0 = zeros(3)
    rf = [11.3; 6.0; 4.5]
    vf = zeros(3)
    qf = Quaternion(deg2rad(0), [0.0; 0.0; 1.0])
    ωf = zeros(3)
    tf_min = 60.0
    tf_max = 200.0
    γ = 0.0
    hom = 50.0
    ε_sdf = 1e-4
    traj = FreeFlyerTrajectoryParameters(
        r0, rf, v0, vf, q0, qf, ω0, ωf, tf_min,
        tf_max, γ, hom, ε_sdf)

    mdl = FreeFlyerProblem(fflyer, env, traj)

    return mdl
end
