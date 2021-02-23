#= General trajectory problem data structures and methods.

This file stores the __general__ data structures and methods which define the
particular instance of the trajectory generation problem. =#

include("../utils/types.jl")

# ..:: Data structures ::..

#= Trajectory problem definition. =#
mutable struct TrajectoryProblem
    # >> Variable sizes <<
    nx::T_Int         # Number of state variables
    nu::T_Int         # Number of input variables
    np::T_Int         # Number of parameter variables
    # >> Initial guess <<
    guess::T_Function # The initial trajectory guess
    # >> Cost function <<
    φ::T_Function     # Terminal cost
    Γ::T_Function     # Running cost
    # >> Dynamics <<
    f::T_Function     # State time derivative
    A::T_Function     # Jacobian df/dx
    B::T_Function     # Jacobian df/du
    F::T_Function     # Jacobian df/dp
    # >> Constraints <<
    X!::T_Function    # Vector of convex state constraints
    U!::T_Function    # Vector of convex input constraints
    s::T_Function     # Nonconvex inequality constraint function
    C::T_Function     # Jacobian ds/dx
    D::T_Function     # Jacobian ds/du
    G::T_Function     # Jacobian ds/dp
    # >> Boundary conditions <<
    gic::T_Function   # Initial condition
    H0::T_Function    # Jacobian dgic/dx
    K0::T_Function    # Jacobian dgic/dp
    gtc::T_Function   # Terminal condition
    Hf::T_Function    # Jacobian dgtc/dx
    Kf::T_Function    # Jacobian dgtc/dp
    # >> Other <<
    mdl::Any          # Problem-specific data structure
end

# ..:: Constructors ::..

#= Default (empty) constructor of a trajectory problem.

Args:
    mdl: problem-specific data.

Returns:
    pbm: an empty trajectory problem. =#
function TrajectoryProblem(mdl::Any)::TrajectoryProblem

    nx = 0
    nu = 0
    np = 0
    guess = (N) -> (T_RealMatrix[], T_RealMatrix[], T_RealVector[])
    φ = (x, p) -> 0.0
    Γ = (x, u, p) -> 0.0
    f = (τ, x, u, p) -> T_RealVector[]
    A = (τ, x, u, p) -> T_RealMatrix[]
    B = (τ, x, u, p) -> T_RealMatrix[]
    F = (τ, x, u, p) -> T_RealMatrix[]
    X! = (x, mdl) -> T_ConstraintVector[]
    U! = (x, mdl) -> T_ConstraintVector[]
    s = (x, u, p) -> T_RealVector[]
    C = (x, u, p) -> T_RealMatrix[]
    D = (x, u, p) -> T_RealMatrix[]
    G = (x, u, p) -> T_RealMatrix[]
    gic = (x, p) -> T_RealVector[]
    H0 = (x, p) -> T_RealMatrix[]
    K0 = (x, p) -> T_RealMatrix[]
    gtc = (x, p) -> T_RealVector[]
    Hf = (x, p) -> T_RealMatrix[]
    Kf = (x, p) -> T_RealMatrix[]

    pbm = TrajectoryProblem(nx, nu, np, guess, φ, Γ, f, A, B, F, X!, U!,
                            s, C, D, G, gic, H0, K0, gtc, Hf, Kf, mdl)

    return pbm
end

# ..:: Methods ::..

#= Set the problem dimensions.

Args:
    pbm: the trajectory problem structure.
    nx: state dimension.
    nu: input dimension.
    np: parameter dimension. =#
function problem_set_dims!(pbm::TrajectoryProblem,
                           nx::T_Int,
                           nu::T_Int,
                           np::T_Int)::Nothing
    pbm.nx = nx
    pbm.nu = nu
    pbm.np = np
    return nothing
end

#= Define the initial trajectory guess.

Function signature: f(pbm, N), where:
  - N (T_Int): the number of discrete-time grid nodes.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return the tuple (x,u,p), where:
  - x (T_RealMatrix): the state trajectory guess.
  - u (T_RealMatrix): the input trajectory guess.
  - p (T_RealVector): the parameter vector.

Args:
    pbm: the trajectory problem structure.
    guess: the guess generator. =#
function problem_set_guess!(pbm::TrajectoryProblem,
                            guess::T_Function)::Nothing
    pbm.guess = (N) -> guess(N, pbm)
    return nothing
end

#= Define the cost function.

Function signature: φ(x, p, pbm), where:
  - x (T_OptiVarVector): the final state.
  - p (T_OptiVarVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

Function signature: Γ(x, u, p, pbm), where:
  - x (T_OptiVarVector): the current state.
  - u (T_OptiVarVector): the current input.
  - p (T_OptiVarVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

Both functions must return a real number.

When you pass "nothing" as the argument, an identically zero function is used
in its place.

Args:
    pbm: the trajectory problem structure.
    φ: the terminal cost.
    Γ: the running cost. =#
function problem_set_cost!(pbm::TrajectoryProblem,
                           φ::T_Function,
                           Γ::T_Function)::Nothing
    if !isnothing(φ)
        pbm.φ = (x, p) -> φ(x, p, pbm)
    end
    if !isnothing(Γ)
        pbm.Γ = (x, u, p) -> Γ(x, u, p, pbm)
    end
    return nothing
end

#= Define the dynamics.

Function signature: f(τ, x, u, p, pbm), where:
  - τ (T_Int): the scaled time in [0,1]
  - x (T_RealVector): the current state vector.
  - u (T_RealVector): the current input vector.
  - p (T_RealVector): the current parameter vector.
  - pbm (TrajectorProblem): the trajectory problem structure.

The function f must return a T_RealVector, while A, B, and F must return a
T_RealMatrix.

Args:
    pbm: the trajectory problem structure.
    f: the dynamics function.
    A: Jacobian with respect to the state, df/dx.
    B: Jacobian with respect to the input, df/du.
    F: Jacobian with respect to the parameter, df/dp. =#
function problem_set_dynamics!(pbm::TrajectoryProblem,
                               f::T_Function,
                               A::T_Function,
                               B::T_Function,
                               F::T_Function)::Nothing
    pbm.f = (τ, x, u, p) -> f(τ, x, u, p, pbm)
    pbm.A = (τ, x, u, p) -> A(τ, x, u, p, pbm)
    pbm.B = (τ, x, u, p) -> B(τ, x, u, p, pbm)
    pbm.F = (τ, x, u, p) -> F(τ, x, u, p, pbm)
    return nothing
end

#= Define the convex state constraint set.

Function signature: X!(x, mdl, pbm), where:
  - x (T_OptiVarVector): the state vector.
  - mdl (Model): the JuMP optimization model object.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return a T_ConstraintVector.

Args:
    pbm: the trajectory problem structure.
    X!: the constraints defining the convex state set. =#
function problem_set_X!(pbm::TrajectoryProblem,
                        X!::T_Function)::Nothing
    pbm.X! = (x, mdl) -> X!(x, mdl, pbm)
    return nothing
end

#= Define the convex input constraint set.

Function signature: U!(u, mdl, pbm), where:
  - u (T_OptiVarVector): the input vector.
  - mdl (Model): the JuMP optimization model object.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return a T_ContraintVector.

Args:
    pbm: the trajectory problem structure.
    U!: the constraints defining the convex input set. =#
function problem_set_U!(pbm::TrajectoryProblem,
                        U!::T_Function)::Nothing
    pbm.U! = (u, mdl) -> U!(u, mdl, pbm)
    return nothing
end

#= Define the nonconvex inequality path constraints.

Function signature: f(x, u, p, pbm), where:
  - x (T_RealVector): the state vector.
  - u (T_RealVector): the input vector.
  - p (T_RealVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function s must return a T_RealVector, while C, D, and G must return a
T_RealMatrix.

Args:
    pbm: the trajectory problem structure.
    s: the constraint function.
    C: Jacobian with respect to the state, ds/dx.
    D: Jacobian with respect to the input, ds/du.
    G: Jacobian with respect to the parameter, ds/dp. =#
function problem_set_s!(pbm::TrajectoryProblem,
                        s::T_Function,
                        C::T_Function,
                        D::T_Function,
                        G::T_Function)::Nothing
    pbm.s = (x, u, p) -> s(x, u, p, pbm)
    pbm.C = (x, u, p) -> C(x, u, p, pbm)
    pbm.D = (x, u, p) -> D(x, u, p, pbm)
    pbm.G = (x, u, p) -> G(x, u, p, pbm)
    return nothing
end

#= Define the boundary conditions.

Function signature: f(x, p, pbm), where:
  - x (T_RealVector): the state vector.
  - p (T_RealVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function g must return a T_RealVector, while H and K must return a
T_RealMatrix.

Args:
    pbm: the trajectory problem structure.
    kind: either :ic (initial condition) or :tc (terminal condition).
    g: the constraint function.
    H: Jacobian with respect to the state, dg/dx.
    K: Jacobian with respect to the parameter, dg/dp. =#
function problem_set_bc!(pbm::TrajectoryProblem,
                         kind::Symbol,
                         g::T_Function,
                         H::T_Function,
                         K::T_Function)::Nothing
    if kind==:ic
        pbm.gic = (x, p) -> g(x, p, pbm)
        pbm.H0 = (x, p) -> H(x, p, pbm)
        pbm.K0 = (x, p) -> K(x, p, pbm)
    else
        pbm.gtc = (x, p) -> g(x, p, pbm)
        pbm.Hf = (x, p) -> H(x, p, pbm)
        pbm.Kf = (x, p) -> K(x, p, pbm)
    end
    return nothing
end
