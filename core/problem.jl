#= General trajectory problem data structures and methods.

This acts as a "parser" interface to define a particular instance of the
trajectory generation problem.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington),
                   and Autonomous Systems Laboratory (Stanford University)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

include("../utils/types.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Trajectory problem definition. =#
mutable struct TrajectoryProblem
    # >> Variable sizes <<
    nx::T_Int         # Number of state variables
    nu::T_Int         # Number of input variables
    np::T_Int         # Number of parameter variables
    # >> Variable scaling advice <<
    xrg::Vector{Union{Nothing, Tuple{T_Real, T_Real}}} # State bounds
    urg::Vector{Union{Nothing, Tuple{T_Real, T_Real}}} # Input bounds
    prg::Vector{Union{Nothing, Tuple{T_Real, T_Real}}} # Parameter bounds
    # >> Numerical integration <<
    integ_actions::T_SpecialIntegrationActions # Special variable treatment
    # >> Initial guess <<
    guess::T_Function # (SCvx/GuSTO) The initial trajectory guess
    # >> Cost function <<
    φ::T_Function     # (SCvx/GuSTO) Terminal cost
    Γ::T_Function     # (SCvx) Running cost
    S::T_Function     # (GuSTO) Running cost quadratic input penalty
    ∇pS::T_Function   # (GuSTO) Jacobian of S wrt parameter vector
    ℓ::T_Function     # (GuSTO) Running cost input-affine penalty
    ∇xℓ::T_Function   # (GuSTO) Jacobian of ℓ wrt state
    ∇pℓ::T_Function   # (GuSTO) Jacobian of ℓ wrt parameter
    g::T_Function     # (GuSTO) Running cost additive penalty
    ∇xg::T_Function   # (GuSTO) Jacobian of g wrt state
    ∇pg::T_Function   # (GuSTO) Jacobian of g wrt parameter
    g_cvx::T_Bool     # (GuSTO) Indicator if g is convex
    # >> Dynamics <<
    f::T_Function     # State time derivative
    A::T_Function     # Jacobian df/dx
    B::T_Function     # Jacobian df/du
    F::T_Function     # Jacobian df/dp
    # >> Constraints <<
    X::T_ConvexSet    # Convex state constraints
    U::T_ConvexSet    # Convex input constraints
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

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Default (empty) constructor of a trajectory problem.

Args:
    mdl: problem-specific data.

Returns:
    pbm: an empty trajectory problem. =#
function TrajectoryProblem(mdl::Any)::TrajectoryProblem

    nx = 0
    nu = 0
    np = 0
    xrg = Vector{Nothing}(undef, 0)
    urg = Vector{Nothing}(undef, 0)
    prg = Vector{Nothing}(undef, 0)
    propag_actions = T_SpecialIntegrationActions(undef, 0)
    guess = (N) -> (T_RealMatrix(undef, 0, 0),
                    T_RealMatrix(undef, 0, 0),
                    T_RealVector(undef, 0))
    φ = nothing
    Γ = nothing
    S = nothing
    ∇pS = nothing
    ℓ = nothing
    ∇xℓ = nothing
    ∇pℓ = nothing
    g = nothing
    ∇xg = nothing
    ∇pg = nothing
    g_cvx = true
    f = (τ, x, u, p) -> T_RealVector(undef, 0)
    A = (τ, x, u, p) -> T_RealMatrix(undef, 0, 0)
    B = (τ, x, u, p) -> T_RealMatrix(undef, 0, 0)
    F = (τ, x, u, p) -> T_RealMatrix(undef, 0, 0)
    X = nothing
    U = nothing
    s = (x, u, p) -> T_RealVector(undef, 0)
    C = (x, u, p) -> T_RealMatrix(undef, 0, 0)
    D = (x, u, p) -> T_RealMatrix(undef, 0, 0)
    G = (x, u, p) -> T_RealMatrix(undef, 0, 0)
    gic = (x, p) -> T_RealVector(undef, 0)
    H0 = (x, p) -> T_RealMatrix(undef, 0, 0)
    K0 = (x, p) -> T_RealMatrix(undef, 0, 0)
    gtc = (x, p) -> T_RealVector(undef, 0)
    Hf = (x, p) -> T_RealMatrix(undef, 0, 0)
    Kf = (x, p) -> T_RealMatrix(undef, 0, 0)

    pbm = TrajectoryProblem(nx, nu, np, xrg, urg, prg, propag_actions, guess,
                            φ, Γ, S, ∇pS, ℓ, ∇xℓ, ∇pℓ, g, ∇xg, ∇pg, g_cvx, f,
                            A, B, F, X, U, s, C, D, G, gic, H0, K0, gtc, Hf,
                            Kf, mdl)

    return pbm
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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
    pbm.xrg = fill(nothing, nx)
    pbm.urg = fill(nothing, nu)
    pbm.prg = fill(nothing, np)
    return nothing
end

#= Set variable ranges to advise proper scaling.

If no constraint is found that restricts the variable range, then the range
passed manually into here is used.

Args:
    pbm: the trajectory problem structure.
    which: either :state, :input, or :parameter.
    idx: which elements this range applies to.
    rg: the range itself, (min, max). =#
function problem_advise_scale!(pbm::TrajectoryProblem,
                               which::T_Symbol,
                               idx::T_ElementIndex,
                               rg::Tuple{T_Real, T_Real})::Nothing
    if rg[2] < rg[1]
        err = ArgumentError("ERROR: min must be less than max.")
        throw(err)
    end
    map = Dict(:state => :xrg, :input => :urg, :parameter => :prg)
    for i in idx
        getfield(pbm, map[which])[i] = rg
    end
    return nothing
end

#= Define an action on (part of) the state at integration update step.

Action signature: f(x, pbm), where:
  - x (T_RealVector): subset of the state vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

Args:
    pbm: the trajectory problem structure.
    idx: state elements to which the action applies.
    action: the action to do. Receives the subset of the state, and
        returns the updated/correct value. =#
function problem_set_integration_action!(pbm::TrajectoryProblem,
                                         idx::T_ElementIndex,
                                         action::T_Function)::Nothing
    push!(pbm.integ_actions, (idx, (x) -> action(x, pbm)))
    return nothing
end

#= Define the initial trajectory guess.

Function signature: f(N, pbm), where:
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

#= Define the cost function (SCvx variant).

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

When you pass "nothing" as the argument, this term will be interpreted as zero
in the optimization problem.

Args:
    pbm: the trajectory problem structure.
    φ: (optional) the terminal cost.
    Γ: (optional) the running cost. =#
function problem_set_cost!(pbm::TrajectoryProblem;
                           φ::T_Function=nothing,
                           Γ::T_Function=nothing)::Nothing
    pbm.φ = !isnothing(φ) ? (x, p) -> φ(x, p, pbm) : nothing
    pbm.Γ = !isnothing(Γ) ? (x, u, p) -> Γ(x, u, p, pbm) : nothing
    return nothing
end

#= Define the cost function (GuSTO variant).

The running cost is given by:

    u'*S(p)*u+u'*ℓ(x, p)+g(x, p).

Function signatures: φ(x, p, pbm),
                     S(p, pbm),
                     ∇pS(p, pbm),
                     ℓ(x, p, pbm),
                     ∇xℓ(x, p, pbm),
                     ∇pℓ(x, p, pbm),
                     g(x, p, pbm),
                     ∇xg(x, p, pbm),
                     ∇pg(x, p, pbm), where
  - x (T_OptiVarVector): the current state.
  - p (T_OptiVarVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function S must return a positive-semidefinite R^{nu x nu} matrix; the
function ∇pS must return an np-element array of R^{nu x nu} matrices where the
i-th matrix represents the Jacobian of S with respect to the i-th parameter;
the functions φ, ℓ and g must return a real number; the functions ∇xℓ and ∇xg
must return an R^nx vector; and the functions ∇pℓ and ∇pg must return an R^np
vector.

When you pass "nothing" as the argument, this term will be interpreted as zero
in the optimization problem.

Args:
    pbm: the trajectory problem structure.
    φ: (optional) the terminal cost.
    S: (optional) the input quadratic penalty.
    ∇pS: (optional) the input penalty quadratic form Jacobian wrt state.
    ℓ: (optional) the input-affine penalty function.
    ∇xℓ: (optional) the input-affine penalty function Jacobian wrt state.
    ∇pℓ: (optional) the input-affine penalty function Jacobian wrt parameter.
    g: (optional) the additive penalty function.
    ∇xg: (optional) the additive penalty function Jacobian wrt state.
    ∇pg: (optional) the additive penalty function Jacobian wrt parameter. =#
function problem_set_cost!(pbm::TrajectoryProblem;
                           φ::T_Function=nothing,
                           S::T_Function=nothing,
                           ∇pS::T_Function=nothing,
                           ℓ::T_Function=nothing,
                           ∇xℓ::T_Function=nothing,
                           ∇pℓ::T_Function=nothing,
                           g::T_Function=nothing,
                           ∇xg::T_Function=nothing,
                           ∇pg::T_Function=nothing)::Nothing
    pbm.φ = !isnothing(φ) ? (x, p) -> φ(x, p, pbm) : nothing
    pbm.S = !isnothing(S) ? (p) -> S(p, pbm) : nothing
    pbm.∇pS = !isnothing(∇pS) ? (p) -> ∇pS(p, pbm) : nothing
    pbm.ℓ = !isnothing(ℓ) ? (x, p) -> ℓ(x, p, pbm) : nothing
    pbm.∇xℓ = !isnothing(∇xℓ) ? (x, p) -> ∇xℓ(x, p, pbm) : nothing
    pbm.∇pℓ = !isnothing(∇pℓ) ? (x, p) -> ∇pℓ(x, p, pbm) : nothing
    pbm.g = !isnothing(g) ? (x, p) -> g(x, p, pbm) : nothing
    pbm.∇xg = !isnothing(∇xg) ? (x, p) -> ∇xg(x, p, pbm) : nothing
    pbm.∇pg = !isnothing(∇pg) ? (x, p) -> ∇pg(x, p, pbm) : nothing
    if !isnothing(∇xg) || !isnothing(∇pg)
        pbm.g_cvx = false
    end
    return nothing
end

#= Define the dynamics.

Function signature: f(x, u, p, pbm), where:
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
    pbm.f = (x, u, p) -> f(x, u, p, pbm)
    pbm.A = (x, u, p) -> A(x, u, p, pbm)
    pbm.B = (x, u, p) -> B(x, u, p, pbm)
    pbm.F = (x, u, p) -> F(x, u, p, pbm)
    return nothing
end

#= Define the convex state constraint set.

Function signature: X(x, pbm), where:
  - x (T_OptiVarVector): the state vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return an Vector{T_ConvexConeConstraint}.

Args:
    pbm: the trajectory problem structure.
    X: the conic constraints whose intersection defines the convex
       state set. =#
function problem_set_X!(pbm::TrajectoryProblem,
                        X::T_Function)::Nothing
    pbm.X = (x) -> X(x, pbm)
    return nothing
end

#= Define the convex input constraint set.

Function signature: U(u, pbm), where:
  - u (T_OptiVarVector): the input vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return an Vector{T_ConvexConeConstraint}.

Args:
    pbm: the trajectory problem structure.
    U: the conic constraints whose intersection defines the convex
       input set. =#
function problem_set_U!(pbm::TrajectoryProblem,
                        U::T_Function)::Nothing
    pbm.U = (u, mdl) -> U(u, pbm)
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
