#= Discrete linear time-varying system type.

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

export DLTV

""" Discrete-time linear time-varying system, with virtual control. """
mutable struct DLTV
    # x[:,k+1] = ...
    A::RealTensor     # ...  A[:, :, k]*x[:, k]+ ...
    Bm::RealTensor    # ... +Bm[:, :, k]*u[:, k]+ ...
    Bp::RealTensor    # ... +Bp[:, :, k]*u[:, k+1]+ ...
    F::RealTensor     # ... +F[:, :, k]*p+ ...
    r::RealMatrix     # ... +r[:, k]+ ...
    E::RealTensor     # ... +E[:, :, k]*v
    timing::RealTypes # [s] Time taken to discretize

    """
        DLTV(nx, nu, np, nv, N)

    Basic constructor.

    # Arguments
    - `nx`: state dimension.
    - `nu`: input dimension.
    - `np`: parameter dimension.
    - `nv`: virtual control dimension.
    - `N`: the number of discrete time nodes.

    # Returns
    - `dyn`: the dynamics, with empty (undefined) matrices.
    """
    function DLTV(nx::Int, nu::Int, np::Int, nv::Int, N::Int)::DLTV
        A = RealTensor(undef, nx, nx, N - 1)
        Bm = RealTensor(undef, nx, nu, N - 1)
        Bp = RealTensor(undef, nx, nu, N - 1)
        F = RealTensor(undef, nx, np, N - 1)
        r = RealMatrix(undef, nx, N - 1)
        E = RealTensor(undef, nx, nv, N - 1)
        timing = 0.0

        dyn = new(A, Bm, Bp, F, r, E, timing)

        return dyn
    end
end
