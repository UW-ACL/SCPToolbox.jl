#= This file stores the data structures and methods which define the Quadrotor
Obstacle Avoidance numerical example for SCP. =#

include("../utils/types.jl")

# ..:: Data structures ::..

#= Quadrotor vehicle parameters. =#
struct QuadrotorParameters
    id_r::T_IntRange  # Position indices of the state vector
    id_v::T_IntRange  # Velocity indices of the state vector
    id_xt::T_Int      # Index of time dilation state
    id_u::T_IntRange  # Indices of the thrust input vector
    id_σ::T_Int       # Indices of the slack input
    id_pt::T_Int      # Index of time dilation
    u_nrm_max::T_Real # [N] Maximum thrust
    u_nrm_min::T_Real # [N] Minimum thrust
    tilt_max::T_Real  # [rad] Maximum tilt
end

#= Quadrotor flight environment. =#
struct EnvironmentParameters
    g::T_RealVector     # [m/s^2] Gravity vector
    obsN::T_Int         # Number of obstacles
    obsH::T_RealTensor  # Obstacle shapes (ellipsoids)
    obsc::T_RealMatrix  # Obstacle centers
end

#= Trajectory parameters. =#
struct TrajectoryParameters
    r0::T_RealVector # Initial position
    rf::T_RealVector # Terminal position
    v0::T_RealVector # Initial velocity
    vf::T_RealVector # Terminal velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
end

#= Quadrotor trajectory optimization problem parameters all in one. =#
struct QuadrotorProblem
    vehicle::QuadrotorParameters # The ego-vehicle
    env::EnvironmentParameters   # The environment
    traj::TrajectoryParameters   # The trajectory
end

# ..:: Constructors ::..

#= Constructor for the environment.

Args:
    gnrm: gravity vector norm.
    obsiH: array of obstacle shapes (ellipsoids).
    obsc: arrange of obstacle centers.

Returns:
    env: the environment struct. =#
function EnvironmentParameters(
    gnrm::T_Real,
    obsiH::Vector{T_RealMatrix},
    obsc::Vector{T_RealVector})::EnvironmentParameters

    obsN = length(obsiH)
    obsH = cat(obsiH...; dims=3)
    obsc = cat(obsc...; dims=2)

    # Gravity
    g = zeros(3)
    g[end] = -gnrm

    env = EnvironmentParameters(g, obsN, obsH, obsc)

    return env
end

# ..:: Public methods ::..

#= Get the i-th obstacle.

Args:
    i: the obstacle number.
    pbm: the quadrotor trajectory problem parameters.

Returns:
    H: the obtacle shape (ellipsoid).
    c: the obstacle center. =#
function get_obstacle(i::T_Int, pbm::QuadrotorProblem)::Tuple{T_RealMatrix,
                                                              T_RealVector}

    H = pbm.env.obsH[:, :, i]
    c = pbm.env.obsc[:, i]

    return H, c
end
