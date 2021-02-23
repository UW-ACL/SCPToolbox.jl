#= This file stores the data structures and methods which define the Quadrotor
Obstacle Avoidance numerical example for SCP. =#

include("../utils/types.jl")

# ..:: Data structures ::..

#= Free-flyer vehicle parameters. =#
struct FreeFlyerParameters
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_q::T_IntRange # Quaternion indices of the state vector
    id_ω::T_IntRange # Angular velocity indices of the state vector
    id_xt::T_Int     # Index of time dilation state
    id_T::T_IntRange # Thrust indices of the input vector
    id_M::T_IntRange # Torque indicates of the input vector
    id_pt::T_Int     # Index of time dilation
    T_max::T_Real    # [N] Maximum thrust
    M_max::T_Real    # [N*m] Maximum torque
    m::T_Real        # [kg] Mass
    J::T_RealMatrix  # [kg*m^2] Principle moments of inertia matrix
    R::T_Real        # [m] Vehicle radius (spherical representation)
end

#= Trajectory parameters. =#
struct TrajectoryParameters
    r0::T_RealVector # Initial position
    rf::T_RealVector # Terminal position
    v0::T_RealVector # Initial velocity
    vf::T_RealVector # Terminal velocity
    q0::T_Quaternion # Initial attitude
    qf::T_Quaternion # Terminal attitude
    ω0::T_RealVector # Initial angular velocity
    ωf::T_RealVector # Terminal angular velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
end

#= Free-flyer trajectory optimization problem parameters all in one. =#
struct FreeFlyerProblem
    vehicle::FreeFlyerParameters # The ego-vehicle
    traj::TrajectoryParameters   # The trajectory
end
