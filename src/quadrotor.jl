#= This file stores the data structures and methods which define the Quadrotor
Obstacle Avoidance numerical example for SCP. =#

using LinearAlgebra

include("../utils/types.jl")
include("../utils/helper.jl")
include("problem.jl")

# ..:: Data structures ::..

#= Quadrotor vehicle parameters. =#
struct QuadrotorParameters
    nx::T_Int                       # Number of states
    nu::T_Int                       # Number of inputs
    np::T_Int                       # Number of parameters
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
    generic::GenericParameters       # Generic trajectory problem parameters
    vehicle::QuadrotorParameters     # The ego-vehicle
    env::FlightEnvironmentParameters # The environment
    bbox::TrajectoryBoundingBox      # Bounding box for trajectory TODO remove
    # >> Boundary conditions <<
    x0::T_RealVector                 # Initial state boundary condition
    xf::T_RealVector                 # Final state boundary condition
    tf_min::T_Real                   # Shortest possible flight time
    tf_max::T_Real                   # Longest possible flight time
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
    tilt_max::T_Real)::QuadrotorParameters

    # Sizes
    nx = length([id_r; id_v])
    nu = length([id_u; id_σ])
    np = length([id_t])

    quad = QuadrotorParameters(nx, nu, np, id_r, id_v, id_u, id_σ, id_t,
                               u_nrm_max, u_nrm_min, tilt_max)

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

#= Constructor for the overall trajectory problem.

Args:
    quad: the quadrotor vehicle definition.
    env: the environment definition.
    bbox: the trajectory bounding box. TODO remove
    x0: the initial state vector.
    xf: the terminal state vector.
    tf_min: the shortest possible flight time.
    tf_max: the longest possible flight time.

Returns:
    pbm: the overall trajectory problem definition. =#
function QuadrotorTrajectoryProblem(
    veh::QuadrotorParameters,
    env::FlightEnvironmentParameters,
    bbox::TrajectoryBoundingBox,
    x0::T_RealVector,
    xf::T_RealVector,
    tf_min::T_Real,
    tf_max::T_Real)::QuadrotorTrajectoryProblem

    # Parameters
    n_cvx = 4
    n_ncvx = env.obsN+2
    n_ic = veh.nx
    n_tc = veh.nx

    gen = GenericParameters(n_cvx, n_ncvx, n_ic, n_tc)
    pbm = QuadrotorTrajectoryProblem(gen, veh, env, bbox,
                                     x0, xf, tf_min, tf_max)

    return pbm
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
    nx = veh.nx

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
    nx = pbm.vehicle.nx
    nu = pbm.vehicle.nu
    np = pbm.vehicle.np
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

#= Compute Jacobians of the initial boundary condition constraint.

The initial constraint is assumed to be of the form g(x, p)=0.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function initial_bcs(x::T_RealVector,
                     p::T_RealVector, #nowarn
                     pbm::QuadrotorTrajectoryProblem)::Tuple{T_RealVector,
                                                             T_RealMatrix,
                                                             T_RealMatrix}

    # Parameters
    x0 = pbm.x0
    nx = pbm.vehicle.nx
    np = pbm.vehicle.np

    # Compute constraint value and Jacobians
    g = x-x0
    dgdx = I(nx)
    dgdp = zeros(nx, np)

    return g, dgdx, dgdp
end

#= Compute Jacobians of the terminal boundary condition constraint.

The terminal constraint is assumed to be of the form g(x, p)=0.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function terminal_bcs(x::T_RealVector,
                      p::T_RealVector, #nowarn
                      pbm::QuadrotorTrajectoryProblem)::Tuple{T_RealVector,
                                                              T_RealMatrix,
                                                              T_RealMatrix}

    # Parameters
    xf = pbm.xf
    nx = pbm.vehicle.nx
    np = pbm.vehicle.np

    # Compute constraint value and Jacobians
    g = x-xf
    dgdx = I(nx)
    dgdp = zeros(nx, np)

    return g, dgdx, dgdp
end

#= Add convex constraints to the problem at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function mdl_cvx_constraints!(
    xk::T_OptiVarVector, #nowarn
    uk::T_OptiVarVector,
    p::T_OptiVarVector, #nowarn
    mdl::Model,
    pbm::QuadrotorTrajectoryProblem)::T_ConstraintVector

    # Parameters
    nu = pbm.vehicle.nu
    id_u = pbm.vehicle.id_u
    id_σ = pbm.vehicle.id_σ
    u_nrm_max = pbm.vehicle.u_nrm_max
    u_nrm_min = pbm.vehicle.u_nrm_min
    tilt_max = pbm.vehicle.tilt_max

    # Variables
    uuk = uk[id_u]
    σk = uk[id_σ]

    # The constraints
    cvx = T_ConstraintVector(undef, pbm.generic.n_cvx)
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
    n_ncvx = pbm.generic.n_ncvx
    nx = pbm.vehicle.nx
    nu = pbm.vehicle.nu
    np = pbm.vehicle.np
    obsN = pbm.env.obsN
    id_t = pbm.vehicle.id_t
    time_dilation = p[id_t]

    # Initialize values
    s = zeros(n_ncvx)
    dsdx = zeros(n_ncvx, nx)
    dsdu = zeros(n_ncvx, nu)
    dsdp = zeros(n_ncvx, np)

    # Compute values for all obstacles
    for i = 1:obsN
        s[i], dsdx[i, :] = _quadrotor__obstacle_constraint(i, x, pbm)
    end

    # Compute values for time constraint
    s[obsN+1] = time_dilation-pbm.tf_max
    s[obsN+2] = -time_dilation+pbm.tf_min
    dsdp[obsN+1, id_t] = 1.0
    dsdp[obsN+2, id_t] = -1.0

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
    nx = pbm.vehicle.nx
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
