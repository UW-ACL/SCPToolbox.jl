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
    id_xt::T_Int                    # Index of time dilation state
    id_u::T_IntRange                # Indices of the thrust input vector
    id_σ::T_Int                     # Indices of the slack input
    id_pt::T_Int                    # Index of time dilation
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
    id_xt::T_Int,
    id_u::T_IntRange,
    id_σ::T_Int,
    id_pt::T_Int,
    u_nrm_max::T_Real,
    u_nrm_min::T_Real,
    tilt_max::T_Real)::QuadrotorParameters

    # Sizes
    nx = length([id_r; id_v; id_xt])
    nu = length([id_u; id_σ])
    np = length([id_pt])

    quad = QuadrotorParameters(nx, nu, np, id_r, id_v, id_xt, id_u, id_σ,
                               id_pt, u_nrm_max, u_nrm_min, tilt_max)

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
    id_xt = pbm.vehicle.id_xt
    id_pt = pbm.vehicle.id_pt

    # >> Parameter vector <<
    p = [0.5*(pbm.tf_min+pbm.tf_max)]

    # >> State trajectory <<
    x0 = 0.5*(pbm.bbox.init.x.min+pbm.bbox.init.x.max)
    xf = 0.5*(pbm.bbox.trgt.x.min+pbm.bbox.trgt.x.max)
    x0[id_xt] = p[id_pt]
    xf[id_xt] = p[id_pt]
    x_traj = straightline_interpolate(x0, xf, N)

    # >> Input trajectory <<
    u_antigravity = [-g; norm(g)]
    u_traj = straightline_interpolate(u_antigravity, u_antigravity, N)

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
    time_dilation = p[veh.id_pt]

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
    id_xt = pbm.vehicle.id_xt
    id_u = pbm.vehicle.id_u
    id_pt = pbm.vehicle.id_pt

    # Extract variables
    time_dilation = p[id_pt]

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
    F[:, id_pt] = dynamics(pbm, τ, x, u, p)/time_dilation

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
    id_r = pbm.vehicle.id_r
    id_v = pbm.vehicle.id_v
    id_xt = pbm.vehicle.id_xt
    id_pt = pbm.vehicle.id_pt
    time_dilation = p[id_pt]

    # Compute constraint value and Jacobians
    rhs = zeros(nx)
    rhs[id_r] = x0[id_r]
    rhs[id_v] = x0[id_v]
    rhs[id_xt] = time_dilation
    g = x-rhs
    dgdx = I(nx)
    dgdp = zeros(nx, np)
    dgdp[id_xt, id_pt] = -1.0

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
    id_r = pbm.vehicle.id_r
    id_v = pbm.vehicle.id_v
    id_xt = pbm.vehicle.id_xt
    id_pt = pbm.vehicle.id_pt
    time_dilation = p[id_pt]

    # Compute constraint value and Jacobians
    g = x-[xf[id_r]; xf[id_v]; time_dilation]
    dgdx = I(nx)
    dgdp = zeros(nx, np)
    dgdp[id_xt, id_pt] = -1.0

    return g, dgdx, dgdp
end

#= Add convex state constraints to the problem at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function mdl_X!(
    xk::T_OptiVarVector,
    mdl::Model,
    pbm::QuadrotorTrajectoryProblem)::T_ConstraintVector

    # Parameters
    id_xt = pbm.vehicle.id_xt

    # The constraints
    X = T_ConstraintVector(undef, 0)
    push!(X, @constraint(mdl, pbm.tf_min <= xk[id_xt] <= pbm.tf_max))

    return X
end

#= Add convex input constraints to the problem at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function mdl_U!(
    uk::T_OptiVarVector,
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
    U = T_ConstraintVector(undef, 0)
    push!(U, @constraint(mdl, u_nrm_min <= σk))
    push!(U, @constraint(mdl, σk <= u_nrm_max))
    push!(U, @constraint(mdl, vcat(σk, uuk) in MOI.SecondOrderCone(nu)))
    push!(U, @constraint(mdl, σk*cos(tilt_max) <= uuk[3]))

    return U
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
    nx = pbm.vehicle.nx
    nu = pbm.vehicle.nu
    np = pbm.vehicle.np
    obsN = pbm.env.obsN

    # Initialize values
    s = zeros(obsN)
    dsdx = zeros(obsN, nx)
    dsdu = zeros(obsN, nu)
    dsdp = zeros(obsN, np)

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
