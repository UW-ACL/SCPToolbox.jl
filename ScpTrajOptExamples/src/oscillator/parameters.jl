#= Oscillator with deadband data structures.

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

if isdefined(@__MODULE__, :LanguageServer)
    include("../../../ScpTrajOptUtils/src/ScpTrajOptUtils.jl")
    include("../../../ScpTrajOptParser/src/ScpTrajOptParser.jl")
    include("../../../ScpTrajOptSolvers/src/ScpTrajOptSolvers.jl")
    include("../../../ScpTrajOptUtils/src/globals.jl")
    using .ScpTrajOptUtils
    using .ScpTrajOptParser
    using .ScpTrajOptSolvers
end

using ScpTrajOptUtils

const ST = ScpTypes
const RealValue = ST.RealValue
const RealVector = ST.RealVector

""" Oscillator parameters. """
struct OscillatorParameters
    # ..:: Indices ::..
    id_r::Int           # Position (state)
    id_v::Int           # Velocity (state)
    id_aa::Int          # Actual acceleration (input)
    id_ar::Int          # Reference acceleration (input)
    id_l1aa::Int        # Actual acceleration one-norm (input)
    id_l1adiff::Int     # Acceleration difference one-norm (input)
    id_l1r::ST.IntRange # Position one-norm (parameter)
    # ..:: Mechanical parameters ::..
    ζ::RealValue           # Damping ratio
    ω0::RealValue          # [rad/s] Natural frequency
    # ..:: Control parameters ::..
    a_db::RealValue        # [m/s²] Deadband acceleration
    a_max::RealValue       # [m/s²] Maximum acceleration
end # struct

""" Trajectory parameters. """
mutable struct OscillatorTrajectoryParameters
    r0::RealValue # [m] Initial position
    v0::RealValue # [m/s] Initial velocity
    tf::RealValue # [s] Trajectory duration
    κ1::RealValue # Sigmoid homotopy parameter
    κ2::RealValue # Normalization homotopy parameter
    α::RealValue  # Control usage weight
    γ::RealValue  # Control weight for deadband relaxation
end # struct

""" Oscillator trajectory optimization problem parameters all in one. """
struct OscillatorProblem
    vehicle::OscillatorParameters        # The ego-vehicle
    traj::OscillatorTrajectoryParameters # The trajectory
end # struct

"""
    OscillatorProblem(N)

Constructor for the oscillator with actuator deadband problem.

# Arguments
- `N`: the number of discrete time grid nodes.

# Returns
- `mdl`: the problem definition object.
"""
function OscillatorProblem(N::Int)::OscillatorProblem

    # ..:: Oscillator ::..
    # >> Indices <<
    id_r = 1
    id_v = 2
    id_aa = 1
    id_ar = 2
    id_l1aa = 3
    id_l1adiff = 4
    id_l1r = 1:N
    # >> Mechanical parameters <<
    ζ = 0.5
    ω0 = 1.0
    # >> Control parameters <<
    a_db = 0.05
    a_max = 0.3

    oscillator = OscillatorParameters(
        id_r, id_v, id_aa, id_ar, id_l1aa, id_l1adiff, id_l1r, ζ,
        ω0, a_db, a_max)

    # ..:: Trajectory ::..
    r0 = 1.0
    v0 = 0.0
    tf = 10.0
    κ1 = NaN
    κ2 = 1.0
    α = 0.06
    γ = 1e-2

    traj = OscillatorTrajectoryParameters(r0, v0, tf, κ1, κ2, α, γ)

    mdl = OscillatorProblem(oscillator, traj)

    return mdl
end # function
