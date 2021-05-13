#= Methods for discretization of continuous-time dynamics.

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

LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("general.jl")
end

# ..:: Data structures ::..

"""
`DLTV` describes a general discrete linear time-varying system of the form:

```math
x_{k+1} = A_k x_k+\\sum_{i=1}^m B_k^i u_k^i+F_k p+r_k+E_k v_k
```
"""
mutable struct DLTV
    # The discrete-time update matrices
    A::RealTensor
    B::Vector{RealTensor}
    F::RealTensor
    r::RealMatrix
    E::RealTensor
    # Other properties
    method::DiscretizationType # The type of discretization used
    timing::RealTypes          # Time taken to discretize

    """
        DLTV(nx, nu, np, nvc, N, method)

    Basic constructor.

    # Arguments
    - `nx`: state dimension.
    - `nu`: input dimension.
    - `np`: parameter dimension.
    - `nvc`: virtual control dimension.
    - `N`: the number of discrete time nodes.
    - `method`: the type of discretization used.

    # Returns
    - `dyn`: the dynamics, with empty (undefined) matrices.
    """
    function DLTV(nx::Int, nu::Int, np::Int, nvc::Int, N::Int,
                    method::DiscretizationType)::DLTV

        if method==FOH
            A = RealTensor(undef, nx, nx, N-1)
            B = [RealTensor(undef, nx, nu, N-1),
                 RealTensor(undef, nx, nu, N-1)]
            F = RealTensor(undef, nx, np, N-1)
            r = RealMatrix(undef, nx, N-1)
            E = RealTensor(undef, nx, nvc, N-1)
        elseif method==IMPULSE
            A = RealTensor(undef, nx, nx, N-1)
            B = [RealTensor(undef, nx, nu, N-1)]
            F = RealTensor(undef, nx, np, N-1)
            r = RealMatrix(undef, nx, N-1)
            E = RealTensor(undef, nx, nvc, N-1)
        end

        timing = 0.0

        dyn = new(A, B, F, r, E, method, timing)

        return dyn
    end # function
end # struct

"""
`SCPDiscretizationIndices` provides array indices for convenient access during
dynamics discretization. This is used to extract dynamics update matrices from
the propagation vector during the linearized dynamics discretization process.
"""
struct DiscretizationIndices
    x::IntRange         # Indices for state
    A::IntRange         # Indices for A matrix
    B::Vector{IntRange} # Indices for B matrices
    F::IntRange         # Indices for S matrix
    r::IntRange         # Indices for r vector
    E::IntRange         # Indices for E matrix
    length::Int         # Propagation vector total length

    """
        DiscretizationIndices(traj, E, method)

    Indexing arrays from problem definition.

    # Arguments
    - `traj`: the trajectory problem definition.
    - `E`: the dynamics virtual control coefficient matrix.
    - `method`: the type of discretization method used.

    # Returns
    - `idcs`: the indexing array structure.
    """
    function DiscretizationIndices(
        traj::TrajectoryProblem,
        E::RealMatrix,
        method::DiscretizationType)::DiscretizationIndices

        # Parameters
        nx = traj.nx
        nu = traj.nu
        np = traj.np

        # Build the indices
        id_x  = (1:nx)
        id_A  = id_x[end].+(1:nx*nx)
        if method==FOH
            id_B = Vector{IntRange}(undef, 2)
            id_B[1] = id_A[end].+(1:nx*nu)
            id_B[2] = id_B[1][end].+(1:nx*nu)
            id_B_end = id_B[2][end]
        elseif method==IMPULSE
            id_B = Vector{IntRange}(undef, 0)
            id_B_end = id_A[end]
        end
        id_S  = id_B_end.+(1:nx*np)
        id_r  = id_S[end].+(1:nx)
        id_E  = id_r[end].+(1:length(E))
        id_sz = length([id_x; id_A; vcat(id_B...); id_S; id_r; id_E])

        idcs = new(id_x, id_A, id_B, id_S, id_r, id_E, id_sz)

        return idcs
    end # function
end # struct

# ..:: Methods ::..

"""
    discretize!(ref, pbm)

Compute the discrete-time update matrices for a continuous-time dynamics
system, linearized about a reference trajectory. As a byproduct, this
calculates the nonlinear propagation defects.

# Arguments
- `ref`: reference solution about which to discretize.
- `pbm`: the SCP problem definition.
"""
function discretize!(
    ref::SCPSubproblemSolution, pbm::AbstractSCPProblem)::Nothing

    ref.dyn.timing = get_time()
    derivs_assoc = Dict(FOH=>derivs_foh,
                        # IMPULSE=>derivs_impulse
                        )

    # Parameters
    traj = pbm.traj
    nx = traj.nx
    N = pbm.pars.N
    Nsub = pbm.pars.Nsub
    t = pbm.common.t_grid
    iSx = pbm.common.scale.iSx
    derivs = derivs_assoc[pbm.pars.disc_method]

    # Initialization
    idcs = pbm.common.id
    V0 = zeros(idcs.length)
    V0[idcs.A] = vec(I(nx))
    ref.feas = true

    # Propagate individually over each discrete-time interval
    for k = 1:N-1
        # Reset the state initial condition
        V0[idcs.x] = ref.xd[:, k]

        # Integrate
        f = (t, V) -> derivs(t, V, k, pbm, ref)
        t_subgrid = RealVector(LinRange(t[k], t[k+1], Nsub))
        V = rk4(f, V0, t_subgrid; actions=traj.integ_actions)

        # Set the dynamics update matrices encoded in V
        set_update_matrices(ref, V, k, pbm)

        # Take this opportunity to compute the defect, which will be needed
        # later for the trust region update
        xV = V[idcs.x]
        x_next = ref.xd[:, k+1]
        ref.defect[:, k] = x_next-xV
        if norm(iSx*ref.defect[:, k], Inf) > pbm.pars.feas_tol
            ref.feas = false
        end

    end

    ref.dyn.timing = (get_time()-ref.dyn.timing)/1e9

    return nothing
end # function

"""
    derivs_foh(t, V, k, pbm, ref)

Compute concatenanted time derivative vector for first-order hold dynamics
discretization.

# Arguments
- `t`: the time.
- `V`: the current concatenated vector.
- `k`: the discrete time grid interval.
- `pbm`: the SCP problem definition.
- `ref`: the reference trajectory.

# Returns
- `dVdt`: the time derivative of V.
"""
function derivs_foh(t::RealTypes,
                    V::RealVector,
                    k::Int,
                    pbm::AbstractSCPProblem,
                    ref::SCPSubproblemSolution)::RealVector
    # Parameters
    nx = pbm.traj.nx
    t_span = pbm.common.t_grid[k:k+1]

    # Get current values
    idcs = pbm.common.id
    x = V[idcs.x]
    u = linterp(t, ref.ud[:, k:k+1], t_span) #noerr
    p = ref.p
    Phi = reshape(V[idcs.A], (nx, nx))
    σ_m = (t_span[2]-t)/(t_span[2]-t_span[1])
    σ_p = (t-t_span[1])/(t_span[2]-t_span[1])

    # Compute the state time derivative and local linearization
    f = pbm.traj.f(t, k, x, u, p)
    A = pbm.traj.A(t, k, x, u, p)
    B = pbm.traj.B(t, k, x, u, p)
    F = pbm.traj.F(t, k, x, u, p)
    B_m = σ_m*B
    B_p = σ_p*B
    r = f-A*x-B*u-F*p
    E = pbm.common.E

    # Compute the running derivatives for the discrete-time state update
    # matrices
    iPhi = Phi\I(nx)
    dPhidt = A*Phi
    dBmdt = iPhi*B_m
    dBpdt = iPhi*B_p
    dFdt = iPhi*F
    drdt = iPhi*r
    dEdt = iPhi*E

    dVdt = [f; vec(dPhidt); vec(dBmdt); vec(dBpdt);
            vec(dFdt); drdt; vec(dEdt)]

    return dVdt
end # function

"""
    derivs_impulse(t, V, k, pbm, ref)

Compute concatenanted time derivative vector for impulse-input dynamics
discretization.

# Arguments
- `t`: the time.
- `V`: the current concatenated vector.
- `k`: the discrete time grid interval.
- `pbm`: the SCP problem definition.
- `ref`: the reference trajectory.

# Returns
- `dVdt`: the time derivative of V.
"""
function derivs_impulse(t::RealTypes,
                        V::RealVector,
                        k::Int,
                        pbm::AbstractSCPProblem,
                        ref::SCPSubproblemSolution)::RealVector
    # Parameters
    nx = pbm.traj.nx
    nu = pbm.traj.nu

    # Get current values
    idcs = pbm.common.id
    x = V[idcs.x]
    u0 = zeros(nu) # Coasting during the time interval
    p = ref.p
    Phi = reshape(V[idcs.A], (nx, nx))

    # Compute the state time derivative and local linearization
    f = pbm.traj.f(t, k, x, u0, p)
    A = pbm.traj.A(t, k, x, u0, p)
    F = pbm.traj.F(t, k, x, u0, p)
    r = f-A*x-F*p
    E = pbm.common.E

    # Compute the running derivatives for the discrete-time state update
    # matrices
    iPhi = Phi\I(nx)
    dPhidt = A*Phi
    dFdt = iPhi*F
    drdt = iPhi*r
    dEdt = iPhi*E

    dVdt = [f; vec(dPhidt); vec(dFdt); drdt; vec(dEdt)]

    return dVdt
end # function

"""
    set_update_matrices(ref, V, k, pbm)

The the `k`th stage update matrices for the discrete linear time varying system
`sys`, according to the nonlinear propagation result vector `V`.

# Arguments
- `ref`: reference solution for the discretization.
- `V`: the nonlinear propagation vector output.
- `k`: the update stage for which to set the matrices.
- `pbm`: the SCP problem definition.
"""
function set_update_matrices(
    ref::SCPSubproblemSolution, V::RealVector, k::Int,
    pbm::AbstractSCPProblem)::Nothing

    # Parameters
    traj = pbm.traj
    idcs = pbm.common.id
    method = pbm.pars.disc_method
    dyn = ref.dyn
    nx = traj.nx
    nu = traj.nu
    np = traj.np
    sz_E = size(pbm.common.E)

    # Get the raw RK4 results
    AV = V[idcs.A]
    if method==FOH
        BV = [V[idcs_Bi] for idcs_Bi in idcs.B]
    end
    FV = V[idcs.F]
    rV = V[idcs.r]
    EV = V[idcs.E]

    # Extract the discrete-time update matrices for this time interval
    A_k = reshape(AV, (nx, nx))
    if method==FOH
        B_k = [A_k*reshape(BV_i, (nx, nu)) for BV_i in BV]
    elseif method==IMPULSE
        tk = pbm.common.t_grid[k]
        xk = ref.xd[:, k]
        uk = ref.ud[:, k]
        p = ref.p
        Btk = pbm.traj.B(tk, k, xk, uk, p)
        B_k = [A_k*Btk]
    end
    F_k = A_k*reshape(FV, (nx, np))
    r_k = A_k*rV
    if method==IMPULSE
        r_k -= B_k*uk
    end
    E_k = A_k*reshape(EV, sz_E)

    # Save the discrete-time update matrices
    dyn.A[:, :, k] = A_k
    for i = 1:length(B_k)
        dyn.B[i][:, :, k] = B_k[i]
    end
    dyn.F[:, :, k] = F_k
    dyn.r[:, k] = r_k
    dyn.E[:, :, k] = E_k

    return nothing
end # function

"""
    state_update(k, x, u, p, v, dyn)

Impose the `k`th-stage state update constraint for the dynamics `dyn`. By the
`k`th stage, we mean that the update equation updates state ``x_k`` to state
``x_{k+1}``.

# Arguments
- `k`: the stage for which to write the update equation.
- `x`: the state variables.
- `u`: the input variables.
- `p`: the parameter variable vecor.
- `v`: the virtual control variables.
- `dyn`: the dynamics update matrices object.
- `prg`: the conic optimization problem object to which to add the constraint.
"""
function state_update!(k::Int, x::VarArgBlk, u::VarArgBlk, p::VarArgBlk,
                       v::VarArgBlk, dyn::DLTV, prg::ConicProgram)::Nothing

    xk, xkp1 = x[:, k], x[:, k+1]
    vdk = v[:, k]
    A = dyn.A[:, :, k]
    B = [Bi[:, :, k] for Bi in dyn.B]
    F = dyn.F[:, :, k]
    r = dyn.r[:, k]
    E = dyn.E[:, :, k]
    Id = collect(Int, I(length(xk)))

    if dyn.method==FOH
        uk, ukp1 = u[:, k], u[:, k+1]
        @add_constraint(
            prg, ZERO, "dynamics",
            (xkp1, xk, uk, ukp1, p, vdk), begin # Value
                local xkp1, xk, uk, ukp1, p, vdk = arg #noerr
                xkp1-(A*xk+B[1]*uk+B[2]*ukp1+F*p+r+E*vdk)
            end, begin # Jacobians
                if LangServer; local J = Dict(); end
                J[1] = Id
                J[2] = -A
                J[3] = -B[1]
                J[4] = -B[2]
                J[5] = -F
                J[6] = -E
            end)
    elseif dyn.method==IMPULSE
        uk = u[:, k]
        @add_constraint(
            prg, ZERO, "dynamics",
            (xkp1, xk, uk, p, vdk), begin # Value
                local xkp1, xk, uk, p, vdk = arg #noerr
                xkp1-(A*xk+B[1]*uk+F*p+r+E*vdk)
            end, begin # Jacobians
                if LangServer; local J = Dict(); end
                J[1] = Id
                J[2] = -A
                J[3] = -B[1]
                J[4] = -F
                J[5] = -E
            end)
    end

    return nothing
end # function
