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

""" Trajectory problem definition. """
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
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    TrajectoryProblem(mdl)

Default (empty) constructor of a trajectory problem..

# Arguments
- `mdl`: problem-specific data.

# Returns
- `pbm`: an empty trajectory problem.
"""
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

    pbm = TrajectoryProblem(nx, nu, np, xrg, urg, prg, propag_actions, guess,
                            φ, Γ, S, dSdp, ℓ, dℓdx, dℓdp, g, dgdx, dgdp, S_cvx,
                            ℓ_cvx, g_cvx, f, A, B, F, X, U, s, C, D, G, gic,
                            H0, K0, gtc, Hf, Kf, mdl)

    return pbm
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    problem_set_dims!(pbm, nx, nu, np)

Set the problem dimensions.

# Arguments
- `pbm`: the trajectory problem structure.
- `nx`: state dimension.
- `nu`: input dimension.
- `np`: parameter dimension.
"""
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

"""
    problem_advise_scale!(pbm, which, idx, rg)

Set variable ranges to advise proper scaling. This overrides any automatic
variable scaling that may occur.

# Arguments
- `pbm`: the trajectory problem structure.
- `which`: either :state, :input, or :parameter.
- `idx`: which elements this range applies to.
- `rg`: the range itself, (min, max).
"""
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

"""
    problem_set_integration_action!(pbm, idx, action)

Define an action on (part of) the state at integration update step.

# Arguments
- `pbm`: the trajectory problem structure.
- `idx`: state elements to which the action applies.
- `action`: the action to do. Receives the subset of the state, and returns the
  updated/correct value.
"""
function problem_set_integration_action!(pbm::TrajectoryProblem,
                                         idx::T_ElementIndex,
                                         action::T_Function)::Nothing
    push!(pbm.integ_actions, (idx, (x) -> action(x, pbm)))
    return nothing
end

"""
    problem_set_guess!(pbm, guess)

Define the initial trajectory guess.

# Arguments
- `pbm`: the trajectory problem structure.
- `guess`: the guess generator.
"""
function problem_set_guess!(pbm::TrajectoryProblem,
                            guess::T_Function)::Nothing
    pbm.guess = (N) -> guess(N, pbm)
    return nothing
end

"""
    problem_set_terminal_cost!(pbm, φ)

Define the terminal cost.

# Arguments
- `pbm`: the trajectory problem structure.
- `φ`: (optional) the terminal cost.
"""
function problem_set_terminal_cost!(pbm::TrajectoryProblem,
                                    φ::T_Function)::Nothing
    pbm.φ = (x, p) -> φ(x, p, pbm)
    return nothing
end

"""
    problem_set_running_cost!(pbm, algo, SΓ
                              [, dSdp, ℓ, dℓdx, dℓdp, g, dgdx, dgdp])

Define the running cost function. SCvx just requires the first function, `Γ(x,
u, p)`. GuSTO requires all the arguments and their Jacobians.

Args:
- `pbm`: the trajectory problem structure.
- `algo`: which algorithm is being used.
- `SΓ`: (optional) the running cost is SCvx, or the input quadratic penalty if
  GuSTO.
- `dSdp`: (optional) the input penalty quadratic form Jacobian wrt state.
- `ℓ`: (optional) the input-affine penalty function.
- `dℓdx`: (optional) the input-affine penalty function Jacobian wrt state.
- `dℓdp`: (optional) the input-affine penalty function Jacobian wrt parameter.
- `g`: (optional) the additive penalty function.
- `dgdx`: (optional) the additive penalty function Jacobian wrt state.
- `dgdp`: (optional) the additive penalty function Jacobian wrt parameter.
"""
function problem_set_running_cost!(pbm::TrajectoryProblem,
                                   algo::T_Symbol,
                                   SΓ::T_Function,
                                   dSdp::T_Function=nothing,
                                   ℓ::T_Function=nothing,
                                   dℓdx::T_Function=nothing,
                                   dℓdp::T_Function=nothing,
                                   g::T_Function=nothing,
                                   dgdx::T_Function=nothing,
                                   dgdp::T_Function=nothing)::Nothing
    if algo==:scvx
        pbm.Γ = (x, u, p) -> SΓ(x, u, p, pbm)
    else
        pbm.S = !isnothing(SΓ) ? (p) -> SΓ(p, pbm) : nothing
        pbm.dSdp = !isnothing(dSdp) ? (p) -> dSdp(p, pbm) : nothing
        pbm.S_cvx = isnothing(dSdp)
        pbm.ℓ = !isnothing(ℓ) ? (x, p) -> ℓ(x, p, pbm) : nothing
        pbm.dℓdx = !isnothing(dℓdx) ? (x, p) -> dℓdx(x, p, pbm) : nothing
        pbm.dℓdp = !isnothing(dℓdp) ? (x, p) -> dℓdp(x, p, pbm) : nothing
        pbm.ℓ_cvx = isnothing(dℓdx) && isnothing(dℓdp)
        pbm.g = !isnothing(g) ? (x, p) -> g(x, p, pbm) : nothing
        pbm.dgdx = !isnothing(dgdx) ? (x, p) -> dgdx(x, p, pbm) : nothing
        pbm.dgdp = !isnothing(dgdp) ? (x, p) -> dgdp(x, p, pbm) : nothing
        pbm.g_cvx = isnothing(dgdx) && isnothing(dgdp)
    end
    return nothing
end

"""
    problem_set_dynamics!(pbm, f, A, B, F)

Define the dynamics (SCvx).

# Arguments
- `pbm`: the trajectory problem structure.
- `f`: the dynamics function.
- `A`: Jacobian with respect to the state, `df/dx`.
- `B`: Jacobian with respect to the input, `df/du`.
- `F`: Jacobian with respect to the parameter, `df/dp`.
"""
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

"""
    problem_set_dynamics!(pb, f, A, F)

Define the input-affine dynamics (GuSTO).

# Arguments
- `pbm`: the trajectory problem structure.
- `kind`: either :nonlinear or :inputaffine.
- `f`: the dynamics functions `{f0, f1, ...}`.
- `A`: Jacobians with respect to the state, `{df0/dx, df1/dx, ...}`.
- `F`: Jacobians with respect to the parameter, `{df0/dp, df1/dp, ...}`.
"""
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

"""
    problem_set_X!(pbm, X)

Define the convex state constraint set.

# Arguments
- `pbm`: the trajectory problem structure.
- `X`: the conic constraints whose intersection defines the convex state set.
"""
function problem_set_X!(pbm::TrajectoryProblem,
                        X::T_Function)::Nothing
    pbm.X = (t, k, x, p) -> X(t, k, x, p, pbm)
    return nothing
end

"""
    problem_set_U!(pbm, U)

Define the convex input constraint set.

# Arguments
- `pbm`: the trajectory problem structure.
- `U`: the conic constraints whose intersection defines the convex input set.
"""
function problem_set_U!(pbm::TrajectoryProblem,
                        U::T_Function)::Nothing
    pbm.U = (t, k, u, p) -> U(t, k, u, p, pbm)
    return nothing
end

"""
    problem_set_s!(pbm, algo, s, C, DG, G)

Define the nonconvex inequality path constraints. The SCvx algorithm assumes
the function form `s(t, k, x, u, p)`. The GuSTO algorithm assumes the function
form `s(t, k, x, p)`. Thus, SCvx requires the `s` argument as well as all three
Jacobians. GuSTO requires `s` and the two Jacobians.

Args:
- `pbm`: the trajectory problem structure.
- `algo`: which algorithm is being used.
- `s`: the constraint function.
- `C`: (optional) Jacobian with respect to the state, `ds/dx`.
- `DG`: (optional) Jacobian with respect to the input or parameter, `ds/du` or
  `ds/dp`. If SCvx, `ds/du` is used. If GuSTO, `ds/do` is used.
- `G`: (optional) Jacobian with respect to the parameter, `ds/dp`. Only provide
  if using SCvx.
"""
function problem_set_s!(pbm::TrajectoryProblem,
                        algo::T_Symbol,
                        s::T_Function,
                        C::T_Function=nothing,
                        DG::T_Function=nothing,
                        G::T_Function=nothing)::Nothing
    if isnothing(s)
        err = SCPError(0, SCP_BAD_ARGUMENT, "ERROR: must at least provide s.")
        throw(err)
    end

    not = !isnothing

    if algo==:scvx
        pbm.s = (t, k, x, u, p) -> s(t, k, x, u, p, pbm)
        pbm.C = not(C) ? (t, k, x, u, p) -> C(t, k, x, u, p, pbm) : nothing
        pbm.D = not(DG) ? (t, k, x, u, p) -> DG(t, k, x, u, p, pbm) : nothing
        pbm.G = not(G) ? (t, k, x, u, p) -> G(t, k, x, u, p, pbm) : nothing
    else
        pbm.s = (t, k, x, p) -> s(t, k, x, p, pbm)
        pbm.C = not(C) ? (t, k, x, p) -> C(t, k, x, p, pbm) : nothing
        pbm.G = not(DG) ? (t, k, x, p) -> DG(t, k, x, p, pbm) : nothing
    end

    return nothing
end

"""
    problem_set_bc!(pbm, kind, g, H, K)

Define the boundary conditions.

# Arguments
- `pbm`: the trajectory problem structure.
- `kind`: either :ic (initial condition) or :tc (terminal condition).
- `g`: the constraint function.
- `H`: Jacobian with respect to the state, dg/dx.
- `K`: Jacobian with respect to the parameter, dg/dp.
"""
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
