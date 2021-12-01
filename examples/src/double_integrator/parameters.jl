#= Lossless convexification double integrator data structures.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

using Utils

# ..:: Globals ::..

const RealValue = Types.RealTypes
const RealVector = Types.RealVector
const RealMatrix = Types.RealMatrix

# ..:: Data structures ::..

"""
`DoubleIntegrator` holds the double integrator problem definition parameters.
"""
struct DoubleIntegratorParameters
    n::Int         # Number of states
    m::Int         # Number of inputs
    N::Int         # Number of discrete time grid nodes
    T::RealValue   # Trajectory duration
    f::Function    # Equations of motiong as a function
    A::RealMatrix  # Discrete-time update state coefficients
    Bm::RealVector # Discrete-time update current input coefficients
    Bp::RealVector # Discrete-time update next input coefficients
    w::RealVector  # Discrete-time update exogenous constant disturbance
    g::RealValue   # Friction amount
    s::RealValue   # Travel distance
    choice::Int    # Problem parameters choice
end # struct

"""
`Solution` stores the problem optimal solution.
"""
struct Solution
    t::RealVector # The discrete times at which solution is stored
    x::RealMatrix # The state trajectory (columns for each time)
    u::RealMatrix # The input trajectory (columns for each time)
end # struct

# ..:: Methods ::..


function DoubleIntegratorParameters(
    choice::Int, T::RealValue=10)::DoubleIntegratorParameters

    @assert choice in (1, 2)

    # Parameters
    T = T
    N = 50 # Number of discretization time grid nodes
    A = [0 1;0 0]
    B = [0; 1]
    g = (choice==1) ? 0.1 : 0.6
    s = (choice==1) ? 47 : 30

    # Equations of motion
    f = (t, x, u) -> [x[2]; u-g]

    # First-order hold (FOH) temporal discretization
    n = 2
    m = 1
    dt = T/(N-1)
    t_grid = collect(LinRange(0, dt, 1000))
    Bm = rk4((t, x)->reshape(exp(A*(dt-t))*B*(dt-t)/dt, n*m),
             zeros(n*m), t_grid)
    Bp = rk4((t, x)->reshape(exp(A*(dt-t))*B*t/dt, n*m), zeros(n*m), t_grid)
    w = rk4((t, x)->reshape(exp(A*(dt-t))*[0; -g], n), zeros(n), t_grid)
    A = exp(A*dt)

    mdl = DoubleIntegratorParameters(n, m, N, T, f, A, Bm, Bp, w, g, s, choice)

    return mdl
end
