#= General trajectory problem data structures and methods.

This acts as a "parser" interface to define a particular instance of the
trajectory generation problem.

The following design philosophy applies: you can omit a term by passing
`nothing` to the function. If you leave a piece out of the trajectory problem,
then it is assumed that piece is not present in the optimization problem. For
example, if the running cost is left undefined, then it is taken to be zero.

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
    dSdp::T_Function  # (GuSTO) Jacobian of S wrt parameter vector
    ℓ::T_Function     # (GuSTO) Running cost input-affine penalty
    dℓdx::T_Function  # (GuSTO) Jacobian of ℓ wrt state
    dℓdp::T_Function  # (GuSTO) Jacobian of ℓ wrt parameter
    g::T_Function     # (GuSTO) Running cost additive penalty
    dgdx::T_Function  # (GuSTO) Jacobian of g wrt state
    dgdp::T_Function  # (GuSTO) Jacobian of g wrt parameter
    S_cvx::T_Bool     # (GuSTO) Indicator if S is convex
    ℓ_cvx::T_Bool     # (GuSTO) Indicator if ℓ is convex
    g_cvx::T_Bool     # (GuSTO) Indicator if g is convex
    # >> Dynamics <<
    f::T_Function     # State time derivative
    A::T_Function     # Jacobian df/dx
    B::T_Function     # Jacobian df/du
    F::T_Function     # Jacobian df/dp
    # >> Constraints <<
    X::T_Function     # (SCvx/GuSTO) Convex state constraints
    U::T_Function     # (SCvx/GuSTO) Convex input constraints
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
    scp::Any          # SCP algorithm parameter data structure
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
    guess = nothing
    φ = nothing
    Γ = nothing
    S = nothing
    dSdp = nothing
    ℓ = nothing
    dℓdx = nothing
    dℓdp = nothing
    g = nothing
    dgdx = nothing
    dgdp = nothing
    S_cvx = true
    ℓ_cvx = true
    g_cvx = true
    f = nothing
    A = nothing
    B = nothing
    F = nothing
    X = nothing
    U = nothing
    s = nothing
    C = nothing
    D = nothing
    G = nothing
    gic = nothing
    H0 = nothing
    K0 = nothing
    gtc = nothing
    Hf = nothing
    Kf = nothing
    scp = nothing

    pbm = TrajectoryProblem(nx, nu, np, xrg, urg, prg, propag_actions, guess,
                            φ, Γ, S, dSdp, ℓ, dℓdx, dℓdp, g, dgdx, dgdp, S_cvx,
                            ℓ_cvx, g_cvx, f, A, B, F, X, U, s, C, D, G, gic,
                            H0, K0, gtc, Hf, Kf, mdl, scp)

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

#= Define the terminal cost.

Function signature: φ(x, p, pbm), where:
  - x (T_OptiVarVector): the final state.
  - p (T_OptiVarVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return a real number.

Args:
    pbm: the trajectory problem structure.
    φ: (optional) the terminal cost. =#
function problem_set_terminal_cost!(pbm::TrajectoryProblem,
                                    φ::T_Function)::Nothing
    pbm.φ = (x, p) -> φ(x, p, pbm)
    return nothing
end

#= Define the running cost function (SCvx).

Function signature: Γ(x, u, p, pbm), where:
  - x (T_OptiVarVector): the current state.
  - u (T_OptiVarVector): the current input.
  - p (T_OptiVarVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return a real number.

Args:
    pbm: the trajectory problem structure.
    Γ: (optional) the running cost. =#
function problem_set_running_cost!(pbm::TrajectoryProblem,
                                   Γ::T_Function)::Nothing
    pbm.Γ = (x, u, p) -> Γ(x, u, p, pbm)
    return nothing
end

#= Define the cost function (GuSTO variant).

The running cost is given by:

    u'*S(p)*u+u'*ℓ(x, p)+g(x, p).

Function signatures: S(p, pbm),
                     dSdp(p, pbm),
                     ℓ(x, p, pbm),
                     dℓdx(x, p, pbm),
                     dℓdp(x, p, pbm),
                     g(x, p, pbm),
                     dgdx(x, p, pbm),
                     dgdp(x, p, pbm), where
  - x (T_OptiVarVector): the current state.
  - p (T_OptiVarVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The functions must return the following values:
  - S: a positive-semidefinite R^{nu x nu} matrix
  - dSdp: an np-element array of R^{nu x nu} matrices where the i-th matrix
    represents the Jacobian of S with respect to the i-th parameter
  - ℓ: a vector in R^nu
  - dℓdx: a matrix in R^{nu x nx}
  - dℓdp: a matrix in R^{nu x np}
  - g: a real value
  - dgdx: a vector R^nx
  - dgdp: a vector R^np

Args:
    pbm: the trajectory problem structure.
    S: the input quadratic penalty.
    dSdp: the input penalty quadratic form Jacobian wrt state.
    ℓ: the input-affine penalty function.
    dℓdx: the input-affine penalty function Jacobian wrt state.
    dℓdp: the input-affine penalty function Jacobian wrt parameter.
    g: the additive penalty function.
    dgdx: the additive penalty function Jacobian wrt state.
    dgdp: the additive penalty function Jacobian wrt parameter.
    g_cvx: a manual flag whether g(x, p) is a convex function (in which
        case, the return value of g(x, p) must be a convex function in
        JuMP format). =#
function problem_set_running_cost!(pbm::TrajectoryProblem,
                                   S::T_Function,
                                   dSdp::T_Function,
                                   ℓ::T_Function,
                                   dℓdx::T_Function,
                                   dℓdp::T_Function,
                                   g::T_Function,
                                   dgdx::T_Function,
                                   dgdp::T_Function;
                                   g_cvx::T_Bool=false)::Nothing
    pbm.S = !isnothing(S) ? (p) -> S(p, pbm) : nothing
    pbm.dSdp = !isnothing(dSdp) ? (p) -> dSdp(p, pbm) : nothing
    pbm.S_cvx = isnothing(dSdp)
    pbm.ℓ = !isnothing(ℓ) ? (x, p) -> ℓ(x, p, pbm) : nothing
    pbm.dℓdx = !isnothing(dℓdx) ? (x, p) -> dℓdx(x, p, pbm) : nothing
    pbm.dℓdp = !isnothing(dℓdp) ? (x, p) -> dℓdp(x, p, pbm) : nothing
    pbm.ℓ_cvx = isnothing(dℓdx) && isnothing(dℓdp)
    pbm.g = !isnothing(g) ? (x, p) -> g(x, p, pbm) : nothing
    pbm.dgdx = !isnothing(dgdx) ? (x, p) -> dgdx(x, p, pbm) : nothing
    pbm.dgdp = !isnothing(dgdp) ? (x, p) -> dgdp(x, p, pbm) : nothing
    pbm.g_cvx = g_cvx || (isnothing(dgdx) && isnothing(dgdp))
    return nothing
end

#= Define the dynamics (SCvx).

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
    pbm.A = !isnothing(A) ? (x, u, p) -> A(x, u, p, pbm) :
        (x, u, p) -> zeros(pbm.nx, pbm.nx)
    pbm.B = !isnothing(A) ? (x, u, p) -> B(x, u, p, pbm) :
        (x, u, p) -> zeros(pbm.nx, pbm.nu)
    pbm.F = !isnothing(F) ? (x, u, p) -> F(x, u, p, pbm) :
        (x, u, p) -> zeros(pbm.nx, pbm.nu)
    return nothing
end

#= Define the input-affine dynamics (GuSTO).

Function signature: f(x, u, p, pbm), where:
  - x (T_RealVector): the current state vector.
  - u (T_RealVector): the current input vector.
  - p (T_RealVector): the current parameter vector.
  - pbm (TrajectorProblem): the trajectory problem structure.

GuSTO assumes that the dynamics are input-affine, which means:

f(x, u, p) = f0(x, p)+∑_{i=1}^{m} u_i*f_i(x, p)

We thus require the user to provide each function fi (i=1,...,m), and each
functions Jacobian with respect to the state and input. In particular:
  - f must return a Vector{T_RealVector}, the first element of which is taken
    to be independent of the input (i.e, f0 above);
  - A, F must return a Vector{T_RealMatrix}.

Args:
    pbm: the trajectory problem structure.
    kind: either :nonlinear or :inputaffine.
    f: the dynamics functions {f0, f1, ...}.
    A: Jacobians with respect to the state, {df0/dx, df1/dx, ...}.
    F: Jacobians with respect to the parameter, {df0/dp, df1/dp, ...}.. =#
function problem_set_dynamics!(pbm::TrajectoryProblem,
                               f::T_Function,
                               A::T_Function,
                               F::T_Function)::Nothing
    pbm.f = (x, u, p) -> begin
        _f = f(x, p, pbm)
        _f = _f[1]+sum(u[i]*_f[i+1] for i=1:pbm.nu)
        return _f
    end

    pbm.A = !isnothing(A) ? (x, u, p) -> begin
        _A = A(x, p, pbm)
        _A = _A[1]+sum(u[i]*_A[i+1] for i=1:pbm.nu)
        return _A
    end : (x, u, p) -> zeros(pbm.nx, pbm.nx)

    pbm.B = (x, u, p) -> begin
        _B = zeros(pbm.nx, pbm.nu)
        _f = f(x, p, pbm)
        for i = 1:pbm.nu
            _B[:, i] = _f[i+1]
        end
        return _B
    end

    pbm.F = !isnothing(F) ? (x, u, p) -> begin
        _F = F(x, p, pbm)
        _F = _F[1]+sum(u[i]*_F[i+1] for i=1:pbm.nu)
        return _F
    end : (x, u, p) -> zeros(pbm.nx, pbm.nx)

    return nothing
end

#= Define the convex state constraint set.

Function signature: X(t, x, pbm), where:
  - t (T_Real): the current time.
  - x (T_OptiVarVector): the state vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return an Vector{T_ConvexConeConstraint}.

Args:
    pbm: the trajectory problem structure.
    X: the conic constraints whose intersection defines the convex
       state set. =#
function problem_set_X!(pbm::TrajectoryProblem,
                        X::T_Function)::Nothing
    pbm.X = (t, x) -> X(t, x, pbm)
    return nothing
end

#= Define the convex input constraint set.

Function signature: U(t, u, pbm), where:
  - t (T_Real): the current time.
  - u (T_OptiVarVector): the input vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function must return an Vector{T_ConvexConeConstraint}.

Args:
    pbm: the trajectory problem structure.
    U: the conic constraints whose intersection defines the convex
       input set. =#
function problem_set_U!(pbm::TrajectoryProblem,
                        U::T_Function)::Nothing
    pbm.U = (t, u) -> U(t, u, pbm)
    return nothing
end

#= Define the nonconvex inequality path constraints (SCvx version).

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
    if isnothing(s)
        err = SCPError(0, SCP_BAD_ARGUMENT, "ERROR: must at least provide s.")
        throw(err)
    end
    pbm.s = (x, u, p) -> s(x, u, p, pbm)
    pbm.C = !isnothing(C) ? (x, u, p) -> C(x, u, p, pbm) : nothing
    pbm.D = !isnothing(D) ? (x, u, p) -> D(x, u, p, pbm) : nothing
    pbm.G = !isnothing(G) ? (x, u, p) -> G(x, u, p, pbm) : nothing
    return nothing
end

#= Define the nonconvex inequality path constraints (GuSTO version).

Function signature: f(x, u, p, pbm), where:
  - x (T_RealVector): the state vector.
  - p (T_RealVector): the parameter vector.
  - pbm (TrajectoryProblem): the trajectory problem structure.

The function s must return a T_RealVector, while C, D, and G must return a
T_RealMatrix.

Args:
    pbm: the trajectory problem structure.
    s: the constraint function.
    C: Jacobian with respect to the state, ds/dx.
    G: Jacobian with respect to the parameter, ds/dp. =#
function problem_set_s!(pbm::TrajectoryProblem,
                        s::T_Function,
                        C::T_Function,
                        G::T_Function)::Nothing
    if isnothing(s)
        err = SCPError(0, SCP_BAD_ARGUMENT, "ERROR: must at least provide s.")
        throw(err)
    end
    pbm.s = (x, p) -> s(x, p, pbm)
    pbm.C = !isnothing(C) ? (x, p) -> C(x, p, pbm) : nothing
    pbm.G = !isnothing(G) ? (x, p) -> G(x, p, pbm) : nothing
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
    if isnothing(g)
        err = SCPError(0, SCP_BAD_ARGUMENT, "ERROR: must at least provide g.")
        throw(err)
    end
    if kind==:ic
        pbm.gic = (x, p) -> g(x, p, pbm)
        pbm.H0 = !isnothing(H) ? (x, p) -> H(x, p, pbm) : nothing
        pbm.K0 = !isnothing(K) ? (x, p) -> K(x, p, pbm) : nothing
    else
        pbm.gtc = (x, p) -> g(x, p, pbm)
        pbm.Hf = !isnothing(H) ? (x, p) -> H(x, p, pbm) : nothing
        pbm.Kf = !isnothing(K) ? (x, p) -> K(x, p, pbm) : nothing
    end
    return nothing
end
