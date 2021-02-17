#= General trajectory problem data structures and methods.

This file stores the __general__ data structures and methods which define the
particular instance of the trajectory generation problem. =#

include("../utils/types.jl")

# ..:: Data structures ::..

#= Generic properties of the dynamical system. =#
struct GenericDynamicalSystem
    nx::T_Int     # Number of states
    nu::T_Int     # Number of inputs
    np::T_Int     # Number of parameters
    n_cvx::T_Int  # Number of convex inequalities at each time step
    n_ncvx::T_Int # Number of non-convex inequalities at each time step
end

#= Generic bounding box geometric object.

Defines a bounding box, in other words the set:
  {x : min <= x <= max}. =#
struct BoundingBox
    min::T_RealVector # Lower-left vertex
    max::T_RealVector # Upper-right vertex
end

#= Bounding box bounds on the state, input, and time variables. =#
struct XUPBoundingBox
    x::BoundingBox # State bounding box.
    u::BoundingBox # Input bounding box.
    p::BoundingBox # Parameter bounding box.
end

#= Bounding box on the space where the trajectory may lie.

Box bound on the overall trajectory. Composed of a box on the trajectoryy
start, trajectory end, and trajectory middle (i.e., everything in between). =#
struct TrajectoryBoundingBox
    init::XUPBoundingBox # Bounds on the trajectory start.
    trgt::XUPBoundingBox # Bounds on the trajectory end.
    path::XUPBoundingBox # Bounds on everything in between.
end

# ..:: Methods ::..

#= Discrete initial trajectory guess.

Abstract method for computing the discrete initial trajectory guess.

Args:
    pbm: the trajectory problem description.
    N: the number of discrete points along the trajectory to output.

Returns:
    x_traj: the state trajectory.
    u_traj: the input trajectory.
    p: the parameter vector. =#
function initial_guess(
    pbm::T, #nowarn
    N::T_Int)::Tuple{T_RealMatrix, #nowarn
                     T_RealMatrix,
                     T_RealVector} where {T<:AbstractTrajectoryProblem}

    x_traj = T_RealMatrix(undef, pbm.vehicle.generic.nx, N)
    u_traj = T_RealMatrix(undef, pbm.vehicle.generic.nu, N)
    p = T_RealVector(undef, pbm.vehicle.generic.np)

    return x_traj, u_traj, p
end

#= State time derivative.

Abstract method for computing the state time derivative.

Args:
    pbm: the trajectory problem description.
    τ: the current (scaled) time.
    x: the current state.
    u: the current input.
    p: the parameter vector.

Returns:
    dxdt: the current state time derivative. =#
function dynamics(
    pbm::T,
    τ::T_Real, #nowarn
    x::T_RealVector, #nowarn
    u::T_RealVector, #nowarn
    p::T_RealVector)::T_RealVector where {T<:AbstractTrajectoryProblem} #nowarn

    nx = pbm.vehicle.generic.nx
    dxdt = T_RealVector(undef, nx)

    return dxdt
end

#= Jacobians of the dynamics.

Abstract method for computing the dynamics' Jacobians with respect to the
state, input and parameters.

Args:
    pbm: the trajectory problem description.
    τ: the current (scaled) time.
    x: the current state.
    u: the current input.
    p: the parameter vector.

Returns:
    A: the current Jacobian with respect to the state.
    B: the current Jacobian with respect to the input.
    F: the current Jacobian with respect to the parameters. =#
function jacobians(
    pbm::T,
    τ::T_Real, #nowarn
    x::T_RealVector, #nowarn
    u::T_RealVector, #nowarn
    p::T_RealVector)::Tuple{T_RealMatrix, #nowarn
                            T_RealMatrix,
                            T_RealMatrix} where {T<:AbstractTrajectoryProblem}

    nx = pbm.vehicle.generic.nx
    nu = pbm.vehicle.generic.nu
    np = pbm.vehicle.generic.np

    A = fill(NaN, (nx, nx))
    B = fill(NaN, (nx, nu))
    F = fill(NaN, (nx, np))

    return A, B, F
end

#= Add convex constraints to the problem at time step k.

Args:
    xk: the state vector at time k.
    uk: the input vector at time k.
    p: the parameter vector.
    mdl: the optimization model (JuMP format).
    pbm: the trajectory problem instance.

Returns:
    cvx: vector of convex constraints. =#
function add_mdl_cvx_constraints!(
    xk::T_OptiVarVector, #nowarn
    uk::T_OptiVarVector, #nowarn
    p::T_OptiVarVector, #nowarn
    mdl::Model, #nowarn
    pbm::T)::T_ConstraintVector where {T<:AbstractTrajectoryProblem} #nowarn

    cvx = T_ConstraintVector(undef, 0)

    return cvx
end

#= Get the value and Jacobians of the nonconvex constraints.

Assume that the constraint is of the form s(x, u, p)<=0.

Args:
    x: the state vector.
    u: the input vector.
    p: the parameter vector.
    pbm: the trajectory problem definition.

Returns:
    s: the constraint left-hand side value.
    dsdx: Jacobian with respect to x.
    dsdu: Jacobian with respect to u.
    dsdp: Jacobian with respect to p. =#
function ncvx_constraints(
    x::T_RealVector, #nowarn
    u::T_RealVector, #nowarn
    p::T_RealVector, #nowarn
    pbm::T)::Tuple{T_RealVector,
                   T_RealMatrix,
                   T_RealMatrix,
                   T_RealMatrix} where {T<:AbstractTrajectoryProblem} #nowarn

    n_ncvx = pbm.vehicle.generic.n_ncvx
    nx = pbm.vehicle.generic.nx
    nu = pbm.vehicle.generic.nu
    np = pbm.vehicle.generic.np

    s = T_RealVector(undef, n_ncvx)
    dsdx = T_RealMatrix(undef, n_ncvx, nx)
    dsdu = T_RealMatrix(undef, n_ncvx, nu)
    dsdp = T_RealMatrix(undef, n_ncvx, np)

    return s, dsdx, dsdu, dsdp
end

#= Compute linearized nonconvex constraints at time step k.

Assume that the constraints are of the form s(x, u, p)<=0. This function
outputs only the left-hand side, i.e. the linearization of s().

Args:
    xk: the state vector at time k.
    uk: the input vector at time k.
    p: the parameter vector.
    xbk: reference trajectory state vector at time k.
    ubk: reference trajectory input vector at time k.
    pb: reference trajectory parameter vector.
    pbm: the trajectory problem instance.

Returns:
    lhs: vector of linearized nonconvex constraints left-hand sides. =#
function mdl_ncvx_constraints(
    xk::T_OptiVarVector,
    uk::T_OptiVarVector,
    p::T_OptiVarVector,
    xbk::T_RealVector,
    ubk::T_RealVector,
    pb::T_RealVector,
    pbm::T)::T_OptiVarVector where {T<:AbstractTrajectoryProblem}

    # Parameters
    n_ncvx = pbm.vehicle.generic.n_ncvx

    # The constraints
    ncvx = T_ConstraintVector(undef, n_ncvx)
    s, C, D, G = ncvx_constraints(xbk, ubk, pb, pbm)
    rp = s-C*xbk-D*ubk-G*pb
    lhs = C*xk+D*uk+G*p+rp

    return lhs
end

#= Return the terminal cost expression.

Args:
    xf: state vector at the final time.
    p: parameter vector.
    pbm: the trajectory problem instance.

Returns:
    cost: the terminal cost expression. =#
function terminal_cost(
    xf::T_OptiVarVector, #nowarn
    p::T_OptiVarVector, #nowarn
    pbm::T)::T_Objective where {T<:AbstractTrajectoryProblem} #nowarn

    cost = 0.0

    return cost
end

#= Return the running cost expression at time step k.

Args:
    xk: the state vector at time step k.
    uk: the input vector at time step k.
    p: the parameter vector.
    pbm: the trajectory problem instance.

Returns:
    cost: the running cost expression. =#
function running_cost(
    xk::T_OptiVarVector, #nowarn
    uk::T_OptiVarVector, #nowarn
    p::T_OptiVarVector, #nowarn
    pbm::T)::T_Objective where {T<:AbstractTrajectoryProblem} #nowarn

    cost = 0.0

    return cost
end
