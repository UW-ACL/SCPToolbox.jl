"""
General trajectory problem data structures and methods.

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
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

using LinearAlgebra
using JuMP

using Utils

export TrajectoryProblem
export problem_set_dims!, problem_advise_scale!,
    problem_set_integration_action!, problem_set_guess!,
    problem_set_callback!, problem_set_terminal_cost!,
    problem_set_running_cost!, problem_set_dynamics!, problem_set_X!,
    problem_set_U!, problem_set_s!, problem_set_bc!, problem_add_table_column!,
    define_conic_constraint!
export DiscretizationType, FOH, IMPULSE

# ..:: Globals ::..

@enum(DiscretizationType, FOH, IMPULSE)

const RealTuple = Tuple{Types.RealTypes, Types.RealTypes}
const VectorOfTuples = Vector{Union{Nothing, RealTuple}}
const SIA = Types.SpecialIntegrationActions
const Func = Types.Func
const TableColumnDef = Tuple{Symbol, String, String, Int, Function}

# ..:: Data structures ::..

""" Trajectory problem definition."""
mutable struct TrajectoryProblem
    # >> Variable sizes <<
    nx::Int             # Number of state variables
    nu::Int             # Number of input variables
    np::Int             # Number of parameter variables
    # >> Variable scaling advice <<
    xrg::VectorOfTuples # State bounds
    urg::VectorOfTuples # Input bounds
    prg::VectorOfTuples # Variable parameter bounds
    # >> Numerical integration <<
    integ_actions::SIA  # Special variable treatment
    # >> Initial guess <<
    guess::Func         # (SCvx/GuSTO) The initial trajectory guess
    # >> Callback during solution <<
    callback!::Func     # Callback function after subproblem solution
    # >> Cost function <<
    φ::Func     # (SCvx/GuSTO) Terminal cost
    Γ::Func     # (SCvx) Running cost
    S::Func     # (GuSTO) Running cost quadratic input penalty
    dSdp::Func  # (GuSTO) Jacobian of S wrt parameter vector
    ℓ::Func     # (GuSTO) Running cost input-affine penalty
    dℓdx::Func  # (GuSTO) Jacobian of ℓ wrt state
    dℓdp::Func  # (GuSTO) Jacobian of ℓ wrt parameter
    g::Func     # (GuSTO) Running cost additive penalty
    dgdx::Func  # (GuSTO) Jacobian of g wrt state
    dgdp::Func  # (GuSTO) Jacobian of g wrt parameter
    S_cvx::Bool # (GuSTO) Indicator if S is convex
    ℓ_cvx::Bool # (GuSTO) Indicator if ℓ is convex
    g_cvx::Bool # (GuSTO) Indicator if g is convex
    # >> Dynamics <<
    f::Func     # State time derivative
    A::Func     # Jacobian df/dx
    B::Func     # Jacobian df/du
    F::Func     # Jacobian df/dp
    # >> Constraints <<
    X::Func     # (SCvx/GuSTO) Convex state constraints
    U::Func     # (SCvx/GuSTO) Convex input constraints
    s::Func     # Nonconvex inequality constraint function
    C::Func     # Jacobian ds/dx
    D::Func     # Jacobian ds/du
    G::Func     # Jacobian ds/dp
    ind_X::Union{
        Vector{VariableArgumentBlock},
        Vector{Union{
            Types.RealTypes, Types.RealVector}}} # Indicator variables for convex state constraints
    # >> Boundary conditions <<
    gic::Func   # Initial condition
    H0::Func    # Jacobian dgic/dx
    K0::Func    # Jacobian dgic/dp
    gtc::Func   # Terminal condition
    Hf::Func    # Jacobian dgtc/dx
    Kf::Func    # Jacobian dgtc/dp
    # >> Other <<
    mdl::Any    # Problem-specific data structure
    scp::Any    # SCP algorithm parameter data structure
    force_hard::Bool # Force hard enforcement of constraints
    table_cols::Vector{TableColumnDef} # Extra progress table columns
end # struct

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
    xrg = VectorOfTuples(undef, 0)
    urg = VectorOfTuples(undef, 0)
    prg = VectorOfTuples(undef, 0)
    propag_actions = SIA(undef, 0)
    guess = nothing
    callback! = nothing
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
    ind_X = VariableArgumentBlock[]
    gic = nothing
    H0 = nothing
    K0 = nothing
    gtc = nothing
    Hf = nothing
    Kf = nothing
    scp = nothing
    force_hard = false
    table_cols = TableColumnDef[]

    pbm = TrajectoryProblem(
        nx, nu, np, xrg, urg, prg, propag_actions, guess, callback!, φ, Γ, S, dSdp, ℓ, dℓdx, dℓdp,
        g, dgdx, dgdp, S_cvx, ℓ_cvx, g_cvx, f, A, B, F, X, U, s, C, D, G, ind_X, gic, H0, K0, gtc,
        Hf, Kf, mdl, scp, force_hard, table_cols)

    return pbm
end

# ..:: Methods ::..

"""
    problem_set_dims!(pbm, nx, nu, np[, nq])

Set the problem dimensions.

# Arguments
- `pbm`: the trajectory problem structure.
- `nx`: state dimension.
- `nu`: input dimension.
- `np`: parameter dimension.
"""
function problem_set_dims!(pbm::TrajectoryProblem,
                           nx::Int, nu::Int, np::Int)::Nothing
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
- `which`: either `:state`, `:input`, or `:parameter`.
- `idx`: which elements this range applies to.
- `rg`: the range itself, (min, max).
"""
function problem_advise_scale!(pbm::TrajectoryProblem,
                               which::Symbol,
                               idx::Types.Index,
                               rg::RealTuple)::Nothing
    if rg[2] < rg[1]
        err = ArgumentError("min must be less than max")
        throw(err)
    end
    map = Dict(:state => (pbm.nx, pbm.xrg),
               :input => (pbm.nu, pbm.urg),
               :parameter => (pbm.np, pbm.prg))
    nz, zrg = map[which]
    for i in LinearIndices(1:nz)[idx]
        zrg[i] = rg
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
function problem_set_integration_action!(
    pbm::TrajectoryProblem, idx::Types.Index, action::Func)::Nothing

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
                            guess::Func)::Nothing
    pbm.guess = (N) -> guess(N, pbm)
    return nothing
end

"""
    problem_set_callback!(pbm, cb)

Set a callback function which is called after each subproblem solution and just
prior to making a decision to stop the iterations. By the time this function is
called, the stopping criterion has been evaluated, but not acted upon (pending
the result of this callback). The callback function must satisfy the following
contract:
- At the input: it receives the argument `bay` which is a dictionary which the
  user may freely use to store any information that would like to be accessed
  for post-processing.
- At the input: it receives the argument `subproblem` which is the complete
  data structure for the SCP subproblem. While you can technically modify it,
  be aware of the consequence of tampering with the internals of the solver! It
  is generally **not** recommended to modify `subproblem` during the callback.
- At the input: it receives the argument `mdl` which is the problem-specific
  data object (i.e., the `mdl` field of `TrajectoryProblem`). This is what you
  should modify in the callback, should you have to update anything to guide
  the solution process along.
- At the output: the callback function should return a single boolean flag
  `true` if any value modifications were made inside the function. If so, the
  solver will ignore the stopping criterion and continue iterating. This is
  because the problem definition has changed, so if the current solution
  triggered the stopping criterion, it may no longer do so under the new
  problem parameters.

# Arguments
- `pbm`: the trajectory problem structure.
- `cb`: the callback function.
"""
function problem_set_callback!(pbm::TrajectoryProblem,
                               cb::Func)::Nothing
    pbm.callback! = (subproblem) -> cb(subproblem.sol.bay,
                                       subproblem,
                                       pbm.mdl)
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
                                    φ::Func)::Nothing
    pbm.φ = (x, p) -> φ(x, p, pbm)
    return nothing
end

"""
    problem_set_running_cost!(pbm, algo, SΓ[, dSdp, ℓ, dℓdx,
                              dℓdp, g, dgdx, dgdp])

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
                                   algo::Symbol,
                                   SΓ::Func,
                                   dSdp::Func=nothing,
                                   ℓ::Func=nothing,
                                   dℓdx::Func=nothing,
                                   dℓdp::Func=nothing,
                                   g::Func=nothing,
                                   dgdx::Func=nothing,
                                   dgdp::Func=nothing)::Nothing
    if algo in (:scvx, :ptr)
        pbm.Γ = (t, k, x, u, p) -> SΓ(t, k, x, u, p, pbm)
    else
        pbm.S = !isnothing(SΓ) ? (t, k, p) -> SΓ(t, k, p, pbm) : nothing
        pbm.dSdp = !isnothing(dSdp) ? (t, k, p) -> dSdp(t, k, p, pbm) : nothing
        pbm.S_cvx = isnothing(dSdp)
        pbm.ℓ = !isnothing(ℓ) ? (t, k, x, p) -> ℓ(t, k, x, p, pbm) : nothing
        pbm.dℓdx = !isnothing(dℓdx) ?
            (t, k, x, p) -> dℓdx(t, k, x, p, pbm) : nothing
        pbm.dℓdp = !isnothing(dℓdp) ?
            (t, k, x, p) -> dℓdp(t, k, x, p, pbm) : nothing
        pbm.ℓ_cvx = isnothing(dℓdx) && isnothing(dℓdp)
        pbm.g = !isnothing(g) ?
            (t, k, x, p) -> g(t, k, x, p, pbm) : nothing
        pbm.dgdx = !isnothing(dgdx) ?
            (t, k, x, p) -> dgdx(t, k, x, p, pbm) : nothing
        pbm.dgdp = !isnothing(dgdp) ?
            (t, k, x, p) -> dgdp(t, k, x, p, pbm) : nothing
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
                               f::Func,
                               A::Func,
                               B::Func,
                               F::Func)::Nothing
    pbm.f = (t, k, x, u, p) -> f(t, k, x, u, p, pbm)
    pbm.A = !isnothing(A) ? (t, k, x, u, p) -> A(t, k, x, u, p, pbm) :
        (t, k, x, u, p) -> zeros(pbm.nx, pbm.nx)
    pbm.B = !isnothing(A) ? (t, k, x, u, p) -> B(t, k, x, u, p, pbm) :
        (t, k, x, u, p) -> zeros(pbm.nx, pbm.nu)
    pbm.F = !isnothing(F) ? (t, k, x, u, p) -> F(t, k, x, u, p, pbm) :
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
                               f::Func,
                               A::Func,
                               F::Func)::Nothing
    pbm.f = (t, k, x, u, p) -> begin
        _f = f(t, k, x, p, pbm)
        _f = _f[1]+sum(u[i]*_f[i+1] for i=1:pbm.nu)
        return _f
    end

    pbm.A = !isnothing(A) ? (t, k, x, u, p) -> begin
        _A = A(t, k, x, p, pbm)
        _A = _A[1]+sum(u[i]*_A[i+1] for i=1:pbm.nu)
        return _A
    end : (t, k, x, u, p) -> zeros(pbm.nx, pbm.nx)

    pbm.B = (t, k, x, u, p) -> begin
        _B = zeros(pbm.nx, pbm.nu)
        _f = f(t, k, x, p, pbm)
        for i = 1:pbm.nu
            _B[:, i] = _f[i+1]
        end
        return _B
    end

    pbm.F = !isnothing(F) ? (t, k, x, u, p) -> begin
        _F = F(t, k, x, p, pbm)
        _F = _F[1]+sum(u[i]*_F[i+1] for i=1:pbm.nu)
        return _F
    end : (t, k, x, u, p) -> zeros(pbm.nx, pbm.nx)

    return nothing
end

"""
    problem_set_X!(pbm, X)

Define the convex state constraint set.

# Arguments
- `pbm`: the trajectory problem structure.
- `X`: the conic constraints whose intersection defines the convex state set.
"""
function problem_set_X!(
        pbm::TrajectoryProblem,
        X::Func
)::Nothing
    pbm.X = (ocp, t, k, x, p) -> begin
        mode = (x isa Union{Types.RealTypes, Types.RealVector}) ? :numerical : :optimization
        if mode==:optimization
            pbm.ind_X = Vector{VariableArgumentBlock}(undef, 0)
        else
            pbm.ind_X = Vector{Union{Types.RealTypes, Types.RealVector}}(undef, 0)
        end
        X(t, k, x, p, pbm, ocp)
        return pbm.ind_X
    end
    return nothing
end

"""
    problem_set_U!(pbm, U)

Define the convex input constraint set.

# Arguments
- `pbm`: the trajectory problem structure.
- `U`: the conic constraints whose intersection defines the convex input set.
"""
function problem_set_U!(
        pbm::TrajectoryProblem,
        U::Func
)::Nothing
    pbm.U = (ocp, t, k, u, p) -> begin
        _force_hard = pbm.force_hard
        pbm.force_hard = true
        U(t, k, u, p, pbm, ocp)
        pbm.force_hard = _force_hard
    end
    return nothing
end

"""
    problem_set_s!(pbm, algo, s[, C, DG, G])

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
                        algo::Symbol,
                        s::Func,
                        C::Func=nothing,
                        DG::Func=nothing,
                        G::Func=nothing)::Nothing
    if isnothing(s)
        err = SCPError(0, SCP_BAD_ARGUMENT, "must at least provide s")
        throw(err)
    end

    not = !isnothing

    if algo in (:scvx, :ptr)
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
    problem_set_bc!(pbm, kind, g, H[, K])

Define the boundary conditions.

# Arguments
- `pbm`: the trajectory problem structure.
- `kind`: either :ic (initial condition) or :tc (terminal condition).
- `g`: the constraint function.
- `H`: Jacobian with respect to the state, dg/dx.
- `K`: (optional) Jacobian with respect to the parameter, dg/dp.
"""
function problem_set_bc!(pbm::TrajectoryProblem,
                         kind::Symbol,
                         g::Func,
                         H::Func,
                         K::Func=nothing)::Nothing
    if isnothing(g)
        err = SCPError(0, SCP_BAD_ARGUMENT, "must at least provide g")
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

"""
    problem_add_table_column!(pbm, id, header, format, width, value)

Append a new column to the progress table printout during the SCP iteration
process. The arguments follow the constructor of the `Table` object. The
`value` argument is a function which receives the `bay` field of
`SCPSubproblemSolution`. This is a `Dict` object that stores user-set values in
the callback function, see `problem_set_callback!`. Thus, to have access to
values that you want to display in the table column, you must put them into
`bay` during the callback.

# Arguments
- `pbm`: the trajectory problem structure.
- `id`: a unique symbol referencing the new column.
- `header`: column heading string.
- `format`: column format specifier.
- `width`: column width (number of characters).
- `col_value`: a function with the signature `value(bay)` which receives the
  user-set `bay` field of `SCPSubproblemSolution` (during callback) and returns
  a value appropriate to be `printf`'ed for the column.
"""
function problem_add_table_column!(pbm::TrajectoryProblem,
                                   id::Symbol,
                                   header::String,
                                   format::String,
                                   width::Int,
                                   col_value::Function)::Nothing
    push!(pbm.table_cols, (id, header, format, width, col_value))
    return nothing
end

"""
    define_conic_constraint!(pbm, prog, alg, cone, desc, varlist, definition)

Define a conic constraint. This function abstracts away some complexity in cone constraint
specification for the underlying SCP algorithms that will solve the problem downstream.

For algorithms that require a soft constraint, enforce a "soft" version of the cone by defining a
cone indicator variable. Consider the cone K which defines the constraint x∈K. Let K⊂R^n, an
n-dimensional ambient space. Let q∈R^n be an n-dimensional indicator vector, such that q<=0 implies
x∈K. Furthermore, we formulate q such that if x∈K, then it is feasible to set q<=0. Hence,
effectively, we have a bidirectional relationship: q<=0 if and only if x∈K.

# Arguments
- `pbm`: the trajectory problem structure.
- `prog`: the optimization program with which to associate this constraint.
- `alg`: the SCP algorithm name.
- `cone`: the convex cone that the function value is to lie inside of.
- `desc`: the constraint name.
- `varlist`: tuple of variables that the cone is constructed from.
- `definition`: anonymous function that constructs the affine expression `z` that is to be
  constrained to take values inside the cone.
"""
function define_conic_constraint!(
        pbm::TrajectoryProblem,
        prog::ConicProgram,
        alg::Symbol,
        cone::SupportedCone,
        desc::String,
        varlist::Union{
            NTuple{N, VariableArgumentBlock},
            NTuple{N, Union{Types.RealTypes, Types.RealVector}}},
        definition::Func
)::Nothing where N

    # Check the mode in which to run the function
    mode = (varlist isa NTuple{N, VariableArgumentBlock}) ? :optimization : :numerical

    # Sanitize arguments so that scalars are actually scalars and not zero-dimensional arrays
    scalarize = (z) -> (z isa Array && length(z)==1) ? scalarize(z[1]) : z
    scalarize_args = (args) -> [scalarize(arg) for arg in args]

    if pbm.force_hard || alg != :gusto
        @add_constraint(prog, cone, desc, (varlist...,), definition(scalarize_args(arg)...))
    else
        if mode==:optimization
            if cone in (ZERO, NONPOS)
                _z = definition(scalarize_args(value.(varlist))...)
                cone_dim = (_z isa Array) ? length(_z) : 1
                q = @new_variable(prog, cone_dim, "q")
                @add_constraint(
                    prog, NONPOS, desc, (varlist..., q), begin
                        local arg = scalarize_args(arg)
                        local varlist, q = arg[1:end-1], arg[end]
                        local z = definition(varlist...)
                        z-q
                    end)
                if cone == ZERO
                    @add_constraint(
                        prog, NONPOS, desc, (varlist..., q), begin
                            local arg = scalarize_args(arg)
                            local varlist, q = arg[1:end-1], arg[end]
                            local z = definition(varlist...)
                            -q-z
                        end)
                end
            else
                q = @new_variable(prog, "q")
                if cone in (L1, SOC, LINF)
                    @add_constraint(
                        prog, cone, desc, (varlist..., q), begin
                            local arg = scalarize_args(arg)
                            local varlist, q = arg[1:end-1], arg[end]
                            local z = definition(varlist...)
                            local t = z[1]
                            local x = z[2:end]
                            vcat(t+q, x)
                        end)
                elseif cone==GEOM
                    @add_constraint(
                        prog, cone, desc, (varlist..., q), begin
                            local arg = scalarize_args(arg)
                            local varlist, q = arg[1:end-1], arg[end]
                            local z = definition(varlist...)
                            local t, x = z[1], z[2:end]
                            vcat(x, t-q)
                        end)
                elseif cone==EXP
                    @add_constraint(
                        prog, cone, desc, (varlist..., q), begin
                            local arg = scalarize_args(arg)
                            local varlist, q = arg[1:end-1], arg[end]
                            local z = definition(varlist...)
                            local x, y, w = z
                            vcat(x, y, w+q)
                        end)
                end
            end
        else
            z = definition(scalarize_args(varlist)...)
            if cone==ZERO
                q = abs.(z)
            elseif cone==NONPOS
                q = z
            elseif cone in (L1, SOC, LINF)
                t = z[1]
                x = z[2:end]
                nrm = Dict(L1 => 1, SOC => 2, LINF => Inf)
                q = norm(x, nrm[cone])-t
            elseif cone==GEOM
                t, x = z[1], z[2:end]
                dim = cone.dim-1
                q = t-exp(1/dim*sum(log.(x)))
            elseif cone==EXP
                x, y, w = z
                q = y*exp(x/y)-w
            end
        end
        push!(pbm.ind_X, q)
    end

    return nothing
end
