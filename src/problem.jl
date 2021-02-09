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
    S: the current Jacobian with respect to the parameters. =#
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
    S = fill(NaN, (nx, np))

    return A, B, S
end

#= Add convex constraints to the problem at time step k.

Args:
    k: the current time step.
    x: the state vectors, where x[:, k] is the state at time k.
    u: the input vectors, where u[:, k] is the input at time k.
    p: the parameter vector.
    mdl: the optimization model (JuMP format).
    pbm: the trajectory problem instance.

Returns:
    cvx: vector of convex constraints.
    fit: vector of additional constraints used to fit the JuMP format. =#
function add_mdl_cvx_constraints!(
    k::T_Int, #nowarn
    x::T_OptiVarAffTransfMatrix, #nowarn
    u::T_OptiVarAffTransfMatrix, #nowarn
    p::T_OptiVarAffTransfVector, #nowarn
    mdl::Model, #nowarn
    pbm::T)::Tuple{T_ConstraintVector, #nowarn
                   T_ConstraintVector} where {T<:AbstractTrajectoryProblem}

    cvx = T_ConstraintVector(undef, 0)
    fit = T_ConstraintVector(undef, 0)

    return cvx, fit
end

#= Add nonconvex constraints to the problem at time step k.

Args:
    k: the current time step.
    x: the state vectors, where x[:, k] is the state at time k.
    u: the input vectors, where u[:, k] is the input at time k.
    p: the parameter vector.
    xb: reference trajectory state vectors, where xb[:, k] state at time k.
    ub: reference trajectory input vectors, where ub[:, k] input at time k.
    pb: reference trajectory parameter vector.
    vbk: the virtual control for time step k.
    mdl: the optimization model (JuMP format).
    pbm: the trajectory problem instance.

Returns:
    ncvx: vector of nonconvex constraints.
    fit: vector of additional constraints used to fit the JuMP format. =#
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
    pbm::T)::Tuple{T_ConstraintVector, #nowarn
                   T_ConstraintVector} where {T<:AbstractTrajectoryProblem}

    ncvx = T_ConstraintVector(undef, 0)
    fit = T_ConstraintVector(undef, 0)

    return ncvx, fit
end

#= Return the terminal cost expression.

Args:
    xf: state vector at the final time.
    p: parameter vector.
    mdl: the optimization model (JuMP format).
    pbm: the trajectory problem instance.

Returns:
    cost: the terminal cost expression.
    fit: vector of additional constraints used to fit the JuMP format. =#
function terminal_cost(
    xf::T_OptiVarAffTransfVector, #nowarn
    p::T_OptiVarAffTransfVector, #nowarn
    mdl::Model, #nowarn
    pbm::T)::Tuple{T_Objective, #nowarn
                   T_ConstraintVector} where {
                       T<:AbstractTrajectoryProblem} #nowarn

    cost = 0.0
    fit = T_ConstraintVector(undef, 0)

    return cost, fit
end

#= Return the running cost expression at time step k.

Args:
    k: the current time step.
    x: the state vectors, where x[:, k] is the state at time k.
    u: the input vectors, where u[:, k] is the input at time k.
    p: the parameter vector.
    mdl: the optimization model (JuMP format).
    pbm: the trajectory problem instance.

Returns:
    cost: the running cost expression.
    fit: vector of additional constraints used to fit the JuMP format. =#
function running_cost(
    k::T_Int, #nowarn
    x::T_OptiVarAffTransfMatrix, #nowarn
    u::T_OptiVarAffTransfMatrix, #nowarn
    p::T_OptiVarAffTransfVector, #nowarn
    mdl::Model, #nowarn
    pbm::T)::Tuple{T_Objective, #nowarn
                   T_ConstraintVector} where {
                       T<:AbstractTrajectoryProblem} #nowarn

    cost = 0.0
    fit = T_ConstraintVector(undef, 0)

    return cost, fit
end
