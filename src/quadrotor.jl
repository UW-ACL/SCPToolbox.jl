#= This file stores the data structures and methods which define the Quadrotor
Obstacle Avoidance numerical example for SCP. =#

using LinearAlgebra

include("../utils/types.jl")
include("../utils/helper.jl")
include("problem.jl")

# ..:: Data structures ::..

#= Quadrotor parameters. =#
struct QuadrotorParameters
    generic::GenericDynamicalSystem # Generic parameters
    id_r::T_IntRange                # Position indices of the state vector
    id_v::T_IntRange                # Velocity indices of the state vector
    id_u::T_IntRange                # Indices of the thrust input vector
    id_σ::T_Int                     # Indices of the slack input
    id_t::T_Int                     # Time dilation index
    u_nrm_max::T_Real               # [N] Maximum thrust
    u_nrm_min::T_Real               # [N] Minimum thrust
    tilt_max::T_Real                # [rad] Maximum tilt
end

#= Quadrotor flight environment. =#
struct FlightEnvironmentParameters
    g::T_RealVector     # [m/s^2] Gravity vector
    obsN::T_Int         # Number of obstacles
    obsiH::T_RealTensor # Obstacle shapes (ellipsoids)
    obsc::T_RealMatrix  # Obstacle centers
end

#= Quadrotor trajectory optimization problem parameters all in one. =#
struct QuadrotorTrajectoryProblem<:AbstractTrajectoryProblem
    vehicle::QuadrotorParameters     # The ego-vehicle
    env::FlightEnvironmentParameters # The environment
    bbox::TrajectoryBoundingBox      # Bounding box for trajectory
end

# ..:: Constructors ::..

#= Constructor for the quadrotor vehicle.

Args:
    See the definition of the QuadrotorParameters struct.

Returns:
    quad: the quadrotor vehicle struct. =#
function QuadrotorParameters(
    id_r::T_IntRange,
    id_v::T_IntRange,
    id_u::T_IntRange,
    id_σ::T_Int,
    id_t::T_Int,
    u_nrm_max::T_Real,
    u_nrm_min::T_Real,
    tilt_max::T_Real,
    env::FlightEnvironmentParameters)::QuadrotorParameters

    # Sizes
    nx = length([id_r; id_v])
    nu = length([id_u; id_σ])
    np = length([id_t])
    n_cvx = 4
    n_ncvx = env.obsN

    gen = GenericDynamicalSystem(nx, nu, np, n_cvx, n_ncvx)
    quad = QuadrotorParameters(gen, id_r, id_v, id_u, id_σ, id_t, u_nrm_max,
                               u_nrm_min, tilt_max)

    return quad
end

#= Constructor for the environment.

Args:
    gnrm: gravity vector norm.
    obsiH: array of obstacle shapes (ellipsoids).
    obsc: arrange of obstacle centers.

Returns:
    env: the environment struct. =#
function FlightEnvironmentParameters(
    gnrm::T_Real,
    obsiH::Vector{T_RealMatrix},
    obsc::Vector{T_RealVector})::FlightEnvironmentParameters

    obsN = length(obsiH)
    obsiH = cat(obsiH...;dims=3)
    obsc = cat(obsc...;dims=2)

    # Gravity
    g = zeros(3)
    g[end] = -gnrm

    env = FlightEnvironmentParameters(g, obsN, obsiH, obsc)

    return env
end

# ..:: Public methods ::..

#= Compute a discrete initial guess trajectory.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function initial_guess(
    pbm::QuadrotorTrajectoryProblem,
    N::T_Int)::Tuple{T_RealMatrix,
                     T_RealMatrix,
                     T_RealVector}

    # Parameters
    g = pbm.env.g

    # State trajectory
    x0 = 0.5*(pbm.bbox.init.x.min+pbm.bbox.init.x.max)
    xf = 0.5*(pbm.bbox.trgt.x.min+pbm.bbox.trgt.x.max)
    x_traj = straightline_interpolate(x0, xf, N)

    # Input trajectory
    u_antigravity = [-g; norm(g)]
    u_traj = straightline_interpolate(u_antigravity, u_antigravity, N)

    # >> Parameters <<
    p = 0.5*(pbm.bbox.path.p.min+pbm.bbox.path.p.max)

    return x_traj, u_traj, p
end

#= Compute the state time derivative.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function dynamics(
    pbm::QuadrotorTrajectoryProblem,
    τ::T_Real, #nowarn
    x::T_RealVector,
    u::T_RealVector,
    p::T_RealVector)::T_RealVector

    # Parameters
    veh = pbm.vehicle
    nx = veh.generic.nx

    # Extract variables
    r = x[veh.id_r]
    v = x[veh.id_v]
    uu = u[veh.id_u]
    time_dilation = p[veh.id_t]

    # Individual time derivatives
    drdt = v
    dvdt = uu+pbm.env.g

    # State time derivative
    dxdt = zeros(nx)
    dxdt[veh.id_r] = drdt
    dxdt[veh.id_v] = dvdt
    dxdt *= time_dilation

    return dxdt
end

#= Compute the dynamics' Jacobians with respect to the state and input.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function jacobians(
    pbm::QuadrotorTrajectoryProblem,
    τ::T_Real,
    x::T_RealVector,
    u::T_RealVector,
    p::T_RealVector)::Tuple{T_RealMatrix,
                            T_RealMatrix,
                            T_RealMatrix}

    # Parameters
    nx = pbm.vehicle.generic.nx
    nu = pbm.vehicle.generic.nu
    np = pbm.vehicle.generic.np
    id_r = pbm.vehicle.id_r
    id_v = pbm.vehicle.id_v
    id_u = pbm.vehicle.id_u
    id_t = pbm.vehicle.id_t

    # Extract variables
    time_dilation = p[id_t]

    # Jacobian with respect to the state
    A = zeros(nx, nx)
    A[id_r, id_v] = I(3)
    A *= time_dilation

    # Jacobian with respect to the input
    B = zeros(nx, nu)
    B[id_v, id_u] = I(3)
    B *= time_dilation

    # Jacobian with respect to the parameter vector
    F = T_RealMatrix(undef, nx, np)
    F[:, id_t] = dynamics(pbm, τ, x, u, p)/time_dilation

    return A, B, F
end

#= Add convex constraints to the problem at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function add_mdl_cvx_constraints!(
    xk::T_OptiVarVector, #nowarn
    uk::T_OptiVarVector,
    p::T_OptiVarVector, #nowarn
    mdl::Model,
    pbm::QuadrotorTrajectoryProblem)::T_ConstraintVector

    # Parameters
    nu = pbm.vehicle.generic.nu
    id_u = pbm.vehicle.id_u
    id_σ = pbm.vehicle.id_σ
    u_nrm_max = pbm.vehicle.u_nrm_max
    u_nrm_min = pbm.vehicle.u_nrm_min
    tilt_max = pbm.vehicle.tilt_max

    # Variables
    uuk = uk[id_u]
    σk = uk[id_σ]

    # The constraints
    cvx = T_ConstraintVector(undef, pbm.vehicle.generic.n_cvx)
    cvx[1] = @constraint(mdl, u_nrm_min <= σk)
    cvx[2] = @constraint(mdl, σk <= u_nrm_max)
    cvx[3] = @constraint(mdl, vcat(σk, uuk) in MOI.SecondOrderCone(nu))
    cvx[4] = @constraint(mdl, σk*cos(tilt_max) <= uuk[3])

    return cvx
end

#= Get the value and Jacobians of the nonconvex constraints.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function ncvx_constraints(
    x::T_RealVector,
    u::T_RealVector, #nowarn
    p::T_RealVector, #nowarn
    pbm::QuadrotorTrajectoryProblem)::Tuple{T_RealVector,
                                            T_RealMatrix,
                                            T_RealMatrix,
                                            T_RealMatrix}

    # Parameters
    n_ncvx = pbm.vehicle.generic.n_ncvx
    nx = pbm.vehicle.generic.nx
    nu = pbm.vehicle.generic.nu
    np = pbm.vehicle.generic.np
    obsN = pbm.env.obsN

    # Initialize values
    s = T_RealVector(undef, n_ncvx)
    dsdx = T_RealMatrix(undef, n_ncvx, nx)
    dsdu = zeros(n_ncvx, nu)
    dsdp = zeros(n_ncvx, np)

    # Compute values for all obstacles
    for i = 1:obsN
        s[i], dsdx[i, :] = _quadrotor__obstacle_constraint(i, x, pbm)
    end

    return s, dsdx, dsdu, dsdp
end

#= Return the running cost expression at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function running_cost(
    xk::T_OptiVarVector, #nowarn
    uk::T_OptiVarVector,
    p::T_OptiVarVector, #nowarn
    pbm::QuadrotorTrajectoryProblem)::T_Objective

    # Parameters
    id_σ = pbm.vehicle.id_σ

    # Variables
    σk = uk[id_σ]

    # Running cost value
    cost = σk*σk

    return cost
end

#= Get the i-th obstacle.

Args:
    i: the obstacle number.
    pbm: the trajectory problem definition.

Returns:
    H: the obstacle shape matrix.
    c: the obstacle center. =#
function get_obstacle(i::T_Int,
                      pbm::QuadrotorTrajectoryProblem)::Tuple{T_RealMatrix,
                                                              T_RealVector}
    H = pbm.env.obsiH[:, :, i]
    c  = pbm.env.obsc[:, i]

    return H, c
end

# ..:: Private methods ::..

#= Compute obstacle avoidance constraint and its linearization.

The constraint is of the form f(x)<=0.

Args:
    i: obstacle index.
    xb: vehicle state.
    pbm: the trajectory problem definition.

Returns:
    f: the constraint function f(x) value.
    Df: the Jacobian df/dx. =#
function _quadrotor__obstacle_constraint(
    i::T_Int,
    xb::T_RealVector,
    pbm::QuadrotorTrajectoryProblem)::Tuple{T_Real, T_RealVector}
    # Parameters
    nx = pbm.vehicle.generic.nx
    id_r = pbm.vehicle.id_r
    iH, c = get_obstacle(i, pbm)
    r = xb[id_r]

    # Compute the constraint value f(x)
    tmp = iH*(r-c)
    f = 1-norm(tmp)

    # Compute the constraint linearization at xb
    Df = zeros(nx)
    if norm(tmp)>eps()
        Df[id_r] = -(r-c)'*(iH'*iH)/norm(tmp)
    end

    return f, Df
end
