#= Lossless convexification rocket landing data structures.

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

using LinearAlgebra

using Utils

# ..:: Globals ::..

const RealValue = Types.RealTypes
const RealVector = Types.RealVector
const RealMatrix = Types.RealMatrix

# ..:: Data structures ::..

"""
`Rocket` holds the rocket parameters.
"""
struct Rocket
    g::RealVector     # [m/s²] Acceleration due to gravity
    ω::RealVector     # [rad/s] Planet angular velocity
    m_dry::RealValue  # [kg] Dry mass (structure)
    m_wet::RealValue  # [kg] Wet mass (structure+fuel)
    Isp::RealValue    # [s] Specific impulse
    φ::RealValue      # [rad] Rocket engine cant angle
    α::RealValue      # [s/m] 1/(rocket engine exit velocity)
    ρ_min::RealValue  # [N] Minimum thrust
    ρ_max::RealValue  # [N] Maximum thrust
    γ_gs::RealValue   # [rad] Maximum approach angle
    γ_p::RealValue    # [rad] Maximum pointing angle
    v_max::RealValue  # [m/s] Maximum velocity
    r0::RealVector    # [m] Initial position
    v0::RealVector    # [m] Initial velocity
    Δt::RealValue     # [s] Discretization time step
    A_c::RealMatrix   # Continuous-time dynamics A matrix
    B_c::RealMatrix   # Continuous-time dynamics B matrix
    p_c::RealVector   # Continuous-time dynamics p vector
    n::Int            # Number of states
    m::Int            # Number of inputs
end

"""
`Solution` stores the LCvx solution.
"""
struct Solution
    # >> Raw data <<
    t::RealVector     # [s] Time vector
    r::RealMatrix     # [m] Position trajectory
    v::RealMatrix     # [m/s] Velocity trajectory
    z::RealVector     # [log(kg)] Log(mass) history
    u::RealMatrix     # [m/s^2] Acceleration vector
    ξ::RealVector     # [m/s^2] Acceleration magnitude
    # >> Processed data <<
    cost::RealValue   # Optimization's optimal cost
    T::RealMatrix     # [N] Thrust trajectory
    T_nrm::RealVector # [N] Thrust norm trajectory
    m::RealVector     # [kg] Mass history
    γ::RealVector     # [rad] Pointing angle
end

# ..:: Methods ::..

"""
    Rocket()

Constructor for the rocket.

# Returns
- `rocket`: the rocket definition.
"""
function Rocket()::Rocket

    # >> Basic parameters <<
    e_x = RealVector([1,0,0])
    e_y = RealVector([0,1,0])
    e_z = RealVector([0,0,1])
    g = -3.7114*e_z
    θ = 30*π/180 # [rad] Latitude of landing site
    T_sidereal_mars = 24.6229*3600 # [s]
    ω = (2π/T_sidereal_mars)*(e_x*cos(θ)+e_y*0+e_z*sin(θ))
    m_dry = 1505.0
    m_wet = 1905.0
    Isp = 225.0
    n_eng = 6       # Number of engines
    φ = 27*π/180    # [rad] Engine cant angle off vertical
    T_max = 3.1e3   # [N] Max physical thrust of single engine
    T_1 = 0.3*T_max # [N] Min allowed thrust of single engine
    T_2 = 0.8*T_max # [N] Max allowed thrust of single engine
    ρ_min = n_eng*T_1*cos(φ)
    ρ_max = n_eng*T_2*cos(φ)
    γ_gs = 86*π/180
    γ_p = 40*π/180
    v_max = 500*1e3/3600
    r0 = (2*e_x+0*e_y+1.5*e_z)*1e3
    v0 = 80*e_x+30*e_y-75*e_z
    Δt = 1e0

    # >> Continuous-time dynamics <<
    gₑ = 9.807 # Standard gravity
    α = 1/(Isp*gₑ*cos(φ))
    ω_x = skew(ω)
    A_c = RealMatrix([zeros(3,3) I(3) zeros(3);
                      -(ω_x)^2 -2*ω_x zeros(3);
                      zeros(1,7)])
    B_c = RealMatrix([zeros(3,4);
                      I(3) zeros(3,1);
                      zeros(1,3) -α])
    p_c = RealVector(vcat(zeros(3),g,0))
    n,m = size(B_c)

    # >> Make rocket object <<
    rocket = Rocket(g,ω,m_dry,m_wet,Isp,φ,α,ρ_min,ρ_max,γ_gs,γ_p,v_max,
                    r0,v0,Δt,A_c,B_c,p_c,n,m)

    return rocket
end

"""
    FailedSolution()

Constructor for a failure solution.

# Arguments
- `sol`: a standard "failed" solution.
"""
function FailedSolution()::Solution

    t = RealVector(undef,0)
    r = RealMatrix(undef,0,0)
    v = RealMatrix(undef,0,0)
    z = RealVector(undef,0)
    u = RealMatrix(undef,0,0)
    ξ = RealVector(undef,0)
    cost = Inf
    T = RealMatrix(undef,0,0)
    T_nrm = RealVector(undef,0)
    m = RealVector(undef,0)
    γ = RealVector(undef,0)

    sol = Solution(t,r,v,z,u,ξ,cost,T,T_nrm,m,γ)

    return sol
end
