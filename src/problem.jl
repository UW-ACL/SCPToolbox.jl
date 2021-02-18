#= General trajectory problem data structures and methods.

This file stores the __general__ data structures and methods which define the
particular instance of the trajectory generation problem. =#

include("../utils/types.jl")

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

    x_traj = T_RealMatrix(undef, pbm.vehicle.nx, N)
    u_traj = T_RealMatrix(undef, pbm.vehicle.nu, N)
    p = T_RealVector(undef, pbm.vehicle.np)

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

    nx = pbm.vehicle.nx
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

    nx = pbm.vehicle.nx
    nu = pbm.vehicle.nu
    np = pbm.vehicle.np

    A = fill(NaN, (nx, nx))
    B = fill(NaN, (nx, nu))
    F = fill(NaN, (nx, np))

    return A, B, F
end

#= Compute Jacobians of the initial boundary condition constraint.

The initial constraint is assumed to be of the form g(x, p)=0.

Args:
    x: the state vector at the start time.
    p: the parameter vector.
    pbm: the trajectory problem definition.

Returns:
    g: the initial condition constraint function value.
    dgdx: Jacobian with respect to the state vector.
    dgdp: Jacobian with respect to the parameter vector. =#
function initial_bcs(x::T_RealVector, #nowarn
                     p::T_RealVector, #nowarn
                     pbm::T)::Tuple{T_RealVector, #nowarn
                                    T_RealMatrix,
                                    T_RealMatrix} where {
                                        T<:AbstractTrajectoryProblem}

    # Parameters
    nx = pbm.vehicle.nx
    np = pbm.vehicle.np

    # Jacobians
    g = T_RealVector(undef, 0)
    dgdx = T_RealMatrix(undef, 0, nx)
    dgdp = T_RealMatrix(undef, 0, np)

    return g, dgdx, dgdp
end

#= Compute Jacobians of the terminal boundary condition constraint.

The terminal constraint is assumed to be of the form g(x, p)=0.

Args:
    x: the state vector at the final time.
    p: the parameter vector.
    pbm: the trajectory problem definition.

Returns:
    g: the terminal condition constraint function value.
    dgdx: Jacobian with respect to the state vector.
    dgdp: Jacobian with respect to the parameter vector. =#
function terminal_bcs(x::T_RealVector, #nowarn
                      p::T_RealVector, #nowarn
                      pbm::T)::Tuple{T_RealVector, #nowarn
                                     T_RealMatrix,
                                     T_RealMatrix} where {
                                         T<:AbstractTrajectoryProblem}

    # Parameters
    nx = pbm.vehicle.nx
    np = pbm.vehicle.np

    # Jacobians
    g = T_RealVector(undef, 0)
    dgdx = T_RealMatrix(undef, 0, nx)
    dgdp = T_RealMatrix(undef, 0, np)

    return g, dgdx, dgdp
end

#= Add convex state constraints to the problem at time step k.

Args:
    xk: the state vector at time k.
    mdl: the optimization model (JuMP format).
    pbm: the trajectory problem instance.

Returns:
    X: vector of convex state constraints. =#
function mdl_X!(
    xk::T_OptiVarVector, #nowarn
    mdl::Model, #nowarn
    pbm::T)::T_ConstraintVector where {T<:AbstractTrajectoryProblem} #nowarn

    X = T_ConstraintVector(undef, 0)

    return X
end

#= Add convex input constraints to the problem at time step k.

Args:
    uk: the input vector at time k.
    mdl: the optimization model (JuMP format).
    pbm: the trajectory problem instance.

Returns:
    U: vector of convex state constraints. =#
function mdl_U!(
    uk::T_OptiVarVector, #nowarn
    mdl::Model, #nowarn
    pbm::T)::T_ConstraintVector where {T<:AbstractTrajectoryProblem} #nowarn

    U = T_ConstraintVector(undef, 0)

    return U
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

    nx = pbm.vehicle.nx
    nu = pbm.vehicle.nu
    np = pbm.vehicle.np

    s = T_RealVector(undef, 0)
    dsdx = T_RealMatrix(undef, 0, nx)
    dsdu = T_RealMatrix(undef, 0, nu)
    dsdp = T_RealMatrix(undef, 0, np)

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

    s, C, D, G = ncvx_constraints(xbk, ubk, pb, pbm)
    r = s-C*xbk-D*ubk-G*pb
    lhs = C*xk+D*uk+G*p+r

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
