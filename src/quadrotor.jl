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
    gnrm::T_Real        # Gravity norm
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
    See the definition of the QuadrotorParameters struct.

Returns:
    env: the environment struct. =#
function FlightEnvironmentParameters(
    g::T_Real,
    obsiH::Vector{T_RealMatrix},
    obsc::Vector{T_RealVector})::FlightEnvironmentParameters

    obsN = length(obsiH)
    obsiH = cat(obsiH...;dims=3)
    obsc = cat(obsc...;dims=2)

    env = FlightEnvironmentParameters(g,obsN,obsiH,obsc)

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
    N::T_Int)::Tuple{T_RealMatrix, T_RealMatrix, T_RealVector}

    # >> State trajectory <<
    x0 = 0.5*(pbm.bbox.init.x.min+pbm.bbox.init.x.max)
    xf = 0.5*(pbm.bbox.trgt.x.min+pbm.bbox.trgt.x.max)
    x_traj = straightline_interpolate(x0, xf, N)

    # >> Input trajectory <<
    u0 = pbm.bbox.path.u.min
    uf = pbm.bbox.path.u.min
    u_traj = straightline_interpolate(u0, uf, N)
    # Set first and last input to offset gravity
    g, = _quadrotor__gravity(pbm.env)
    u_traj[:,1] = [-g; norm(g)]
    u_traj[:,N] = [-g; norm(g)]

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

    # Extract variables
    r = x[pbm.vehicle.id_r]
    v = x[pbm.vehicle.id_v]
    u = u[pbm.vehicle.id_u]
    time_dilation = p[pbm.vehicle.id_t]

    # Individual time derivatives
    g, = _quadrotor__gravity(pbm.env)
    drdt = v
    dvdt = u+g

    # State time derivative
    dxdt = zeros(pbm.vehicle.generic.nx)
    dxdt[pbm.vehicle.id_r] = drdt
    dxdt[pbm.vehicle.id_v] = dvdt
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
    τ::T_Real, #nowarn
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
    r = x[id_r]
    v = x[id_v]
    u = u[id_u]
    time_dilation = p[id_t]

    # Jacobian with respect to the state
    _, dgdr = _quadrotor__gravity(pbm.env)
    A = zeros(nx, nx)
    A[id_r, id_v] = I(3)
    A[id_v, id_r] = dgdr
    A *= time_dilation

    # Jacobian with respect to the input
    B = zeros(nx, nu)
    B[id_v, id_u] = I(3)
    B *= time_dilation

    # Jacobian with respect to the parameter vector
    S = T_RealMatrix(undef, nx, np)
    S[:, id_t] = dynamics(pbm, τ, x, u, p)/time_dilation

    return A, B, S
end

#= Add convex constraints to the problem at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function add_mdl_cvx_constraints!(
    k::T_Int,
    x::T_OptiVarAffTransfMatrix, #nowarn
    u::T_OptiVarAffTransfMatrix,
    p::T_OptiVarAffTransfVector, #nowarn
    mdl::Model,
    pbm::QuadrotorTrajectoryProblem)::Tuple{T_ConstraintVector,
                                            T_ConstraintVector}

    # Parameters
    nu = pbm.vehicle.generic.nu
    id_u = pbm.vehicle.id_u
    id_σ = pbm.vehicle.id_σ
    u_nrm_max = pbm.vehicle.u_nrm_max
    u_nrm_min = pbm.vehicle.u_nrm_min
    tilt_max = pbm.vehicle.tilt_max

    # Variables
    uk = u[id_u, k]
    σk = u[id_σ, k]

    # The constraints
    cvx = T_ConstraintVector(undef, pbm.vehicle.generic.n_cvx)
    fit = T_ConstraintVector(undef, 0)
    cvx[1] = @constraint(mdl, u_nrm_min <= σk)
    cvx[2] = @constraint(mdl, σk <= u_nrm_max)
    cvx[3] = @constraint(mdl, vcat(σk, uk) in MOI.SecondOrderCone(nu))
    cvx[4] = @constraint(mdl, σk*cos(tilt_max) <= uk[3])

    return cvx, fit
end

#= Add nonconvex constraints to the problem at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function add_mdl_ncvx_constraint!(
    k::T_Int, #nowarn
    x::T_OptiVarAffTransfMatrix, #nowarn
    u::T_OptiVarAffTransfMatrix, #nowarn
    p::T_OptiVarAffTransfVector, #nowarn
    xb::T_RealMatrix, #nowarn
    ub::T_RealMatrix, #nowarn
    pb::T_RealVector, #nowarn
    vbk::T_OptiVarVector, #nowarn
    mdl::Model, #nowarn
    pbm::QuadrotorTrajectoryProblem)::Tuple{T_ConstraintVector,
                                            T_ConstraintVector}

    # Parameters
    obsN = pbm.env.obsN
    xbk = xb[:, k]

    # Variables
    xk = x[:, k]

    # The constraints
    ncvx = T_ConstraintVector(undef, pbm.vehicle.generic.n_ncvx)
    fit = T_ConstraintVector(undef, 0)
    for i = 1:obsN
        _, a, b = _quadrotor__obstacle_constraint(i, xbk, pbm)
        ncvx[i] = @constraint(mdl, dot(a, xk)+b+vbk[i] <= 0.0)
    end

    return ncvx, fit
end

#= Return the running cost expression at time step k.

Args:
    See docstring of generic method in problem.jl.

Returns:
    See docstring of generic method in problem.jl. =#
function running_cost(
    k::T_Int,
    x::T_OptiVarAffTransfMatrix, #nowarn
    u::T_OptiVarAffTransfMatrix,
    p::T_OptiVarAffTransfVector, #nowarn
    mdl::Model, #nowarn
    pbm::QuadrotorTrajectoryProblem)::Tuple{T_Objective,
                                            T_ConstraintVector}

    # Parameters
    id_σ = pbm.vehicle.id_σ

    # Variables
    σk = u[id_σ, k]

    # Running cost value
    cost = σk*σk
    fit = T_ConstraintVector(undef, 0)

    return cost, fit
end

# ..:: Private methods ::..

#= Get the gravity vector and gravity gradient.

Args:
    env: the environment parameters.

Returns:
    g: the gravity at position r.
    Dg: the gravity gradient at position r. =#
function _quadrotor__gravity(
    env::FlightEnvironmentParameters)::Tuple{T_RealVector,
                                             T_RealMatrix}
    g = [0.0; 0.0; -env.gnrm]
    Dg = zeros(3, 3)
    return g, Dg
end

#= Compute obstacle avoidance constraint and its linearization.

The constraint is of the form f(x)<=0.

Args:
    i: obstacle index.
    xb: vehicle state.
    pbm: the trajectory problem definition.

Returns:
    f: the constraint function f(x) value.
    a: the Jacobian df/dx.
    b: the remainder f-a'*xb. =#
function _quadrotor__obstacle_constraint(
    i::T_Int,
    xb::T_RealVector,
    pbm::QuadrotorTrajectoryProblem)::Tuple{T_Real,
                                            T_RealVector,
                                            T_Real}
    # Parameters
    nx = pbm.vehicle.generic.nx
    id_r = pbm.vehicle.id_r
    iH = pbm.env.obsiH[:, :, i]
    c  = pbm.env.obsc[:, i]
    r = xb[id_r]

    # Compute the constraint value f(x)
    tmp = iH*(r-c)
    f = 1-norm(tmp)

    # Compute the constraint linearization at xb
    a = zeros(nx)
    if norm(tmp)>eps()
        a[id_r] = -(r-c)*dot(iH, iH)/norm(tmp)
    end
    b = f-dot(a, xb)

    return f, a, b
end
