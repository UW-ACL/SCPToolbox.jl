"""
Quadrotor obstacle avoidance data structures and custom methods.

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

""" Quadrotor vehicle parameters. """
struct QuadrotorParameters
    id_r::IntRange # Position indices of the state vector
    id_v::IntRange # Velocity indices of the state vector
    id_u::IntRange # Indices of the thrust input vector
    id_σ::Int      # Index of the slack input
    id_t::Int      # Index of time dilation
    u_max::Real    # [N] Maximum thrust
    u_min::Real    # [N] Minimum thrust
    tilt_max::Real # [rad] Maximum tilt
end

""" Quadrotor flight environment. """
struct QuadrotorEnvironmentParameters
    g::RealVector          # [m/s^2] Gravity vector
    obs::Vector{Ellipsoid} # Obstacles (ellipsoids)
    n_obs::Int             # Number of obstacles
end

""" Trajectory parameters. """
struct QuadrotorTrajectoryParameters
    r0::RealVector # Initial position
    rf::RealVector # Terminal position
    v0::RealVector # Initial velocity
    vf::RealVector # Terminal velocity
    tf_min::Real   # Minimum flight time
    tf_max::Real   # Maximum flight time
    γ::Real        # Minimum-time vs. minimum-energy tradeoff
end

""" Quadrotor trajectory optimization problem parameters all in one. """
struct QuadrotorProblem
    vehicle::QuadrotorParameters        # The ego-vehicle
    env::QuadrotorEnvironmentParameters # The environment
    traj::QuadrotorTrajectoryParameters # The trajectory
end

# ..:: Methods ::..

"""
    QuadrotorEnvironmentParameters(gnrm, obs)

Constructor for the environment.

# Arguments
- `gnrm`: gravity vector norm.
- `obs`: array of obstacles (ellipsoids).

# Returns
- `env`: the environment struct.
"""
function QuadrotorEnvironmentParameters(
        gnrm::Real,
        obs::Vector{Ellipsoid}
)::QuadrotorEnvironmentParameters

    # Derived values
    g = zeros(3)
    g[end] = -gnrm
    n_obs = length(obs)

    env = QuadrotorEnvironmentParameters(g, obs, n_obs)

    return env
end

"""
    QuadrotorProblem()

Constructor for the quadrotor problem.

# Returns
- `mdl`: the quadrotor problem.
"""
function QuadrotorProblem()::QuadrotorProblem

    # >> Quadrotor <<
    id_r = 1:3
    id_v = 4:6
    id_u = 1:3
    id_σ = 4
    id_t = 1
    u_max = 23.2
    u_min = 0.6
    tilt_max = deg2rad(60)
    quad = QuadrotorParameters(id_r, id_v, id_u, id_σ, id_t,
                               u_max, u_min, tilt_max)

    # >> Environment <<
    g = 9.81
    obs = [Ellipsoid(diagm([2.0; 2.0; 0.0]), [1.0; 2.0; 0.0]),
           Ellipsoid(diagm([1.5; 1.5; 0.0]), [2.0; 5.0; 0.0])]
    env = QuadrotorEnvironmentParameters(g, obs)

    # >> Trajectory <<
    r0 = zeros(3)
    rf = zeros(3)
    rf[1:2] = [2.5; 6.0]
    v0 = zeros(3)
    vf = zeros(3)
    tf_min = 0.0
    tf_max = 2.5
    γ = 0.0
    traj = QuadrotorTrajectoryParameters(r0, rf, v0, vf, tf_min, tf_max, γ)

    mdl = QuadrotorProblem(quad, env, traj)

    return mdl
end
