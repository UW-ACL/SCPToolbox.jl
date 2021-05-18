#= Spacecraft rendezvous data structures.

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
    include("../../../utils/src/Utils.jl")

    using .Utils
end

using LinearAlgebra

using Utils

# ..:: Globals ::..

const Ty = Types
const IntRange = Ty.IntRange
const RealValue = Ty.RealTypes
const RealVector = Ty.RealVector
const RealMatrix = Ty.RealMatrix
const Quaternion = Ty.Quaternion
const RCSKey = Tuple{Symbol, Symbol}

# ..:: Data structures ::..

"""
`ApolloCSM` stores the geometry and mechanical properties of the Apollo Command
and Service Module. The values are obtained from [1].

The notation `H_AB` represents a homogeneous transformation from frame `B` to
frame `A`, meaning that `y=H_AB*x` converts the vector `x` in frame `B` to the
corresponding (same) vector `y` in frame `A`.

There are :
- The dynamical frame (D) is the body frame in which the dynamics are
  expressed. It is located at the center of mass. Ultimately, this is the frame
  which we use to impse dynamics in the optimal control problem.
- The structural frame (S) is the frame in which much of Apollo CSM geometry is
  defined. It is located 1000 inches below the command module heat shield main
  structure ablation interface.
- The RCS frame (R) is rotated with respected to the S frame by -(7+15/60)
  degrees about the `x` axis. It is the frame in which the RCS thruster "quads"
  are aligned with respect to the ``\\pm z`` and ``\\pm y`` axes. Otherwise,
  the R and S frames are coincident.
- The quad frame (Q) is a frame located at the center of an RCS quad. There are
  four quads, thus there are four such frames. The frame is positioned such
  that its y axis is normal to the CSM fuselage and its x axis points forward
  (i.e. in the direction of the CSM nose, same as the S frame). The center is
  raises slightly above the CSM fuselage outer mold line, such that the RCS
  thrusters are located in the plane of the Q frame's x and y axes.
  - We refere to the four quads as `:A`, `:B`, `:C`, and `:D`, following NASA
    documentation.
- The thruster frame (T) is a frame centered at the point of application of the
  force produced by an RCS thruster. This is, roughly speaking, in the center
  of the combustion chamber part of the nozzle. The x axis of the T frame
  points out along the nozzle center, and the z axis lies in the (x, y) plane
  of the Q frame. Each thruster is canted 10 degrees away from the CSM
  fuselage. There are four thrusters per quad, hence there is a total of 16 T
  frames.
  - We refer to the thrusters as follows: `:pf` (pitch forward) points along Q
    frame +x, `:pa` (pitch aft) points along Q frame -x, `:rf` (roll forward)
    points along Q frame +z, and `:ra` (roll aft) points along Q frame -z. The
    primary torque effect of firing each thruster is:

                `:A`          `:B`          `:C`          `:D`       (Quad)
            +------------------------------------------------------+
      `:pf` | pitch up      yaw right     pitch down    yaw left   |
      `:pa` | pitch down    yaw left      pitch up      yaw right  |
      `:rf` | roll left     roll left     roll left     roll left  |
      `:ra` | roll right    roll right    roll right    roll right |
            +------------------------------------------------------+
  (Thruster)

- The docking port frame (P) is centered at the docking port in the nose of the
  spacecraft. This is the frame that is to be aligned with the lunar module
  docking port at the docking terminal condition.

References:

[1] CSM/LM Spacecraft Operation Data Book, Volume 3: Mass Properties, National
Aeronautics and Space Administration, SNA-8-D-027(III) REV 2 ed., 1969.
"""
struct ApolloCSM
    # Transformation matrices
    H_SD::RealMatrix
    H_SR::RealMatrix
    H_RQ::Dict{Symbol, RealMatrix}
    H_QT::Dict{Symbol, RealMatrix}
    H_DT::Dict{RCSKey, RealMatrix}
    H_DL::RealMatrix
    # Vectors (in D frame)
    r_rcs::Dict{RCSKey, RealVector} # Thrust application points
    f_rcs::Dict{RCSKey, RealVector} # Thrust vectors
    # Mass properties
    m::RealValue       # [kg] Mass
    J::RealMatrix      # [kg] Inertia matrix in D frame
    # Propulsion properties
    imp_min::RealValue # [N*s] Minimum RCS thruster impulse
    imp_max::RealValue # [N*s] Maximum RCS thruster impulse
    # Other
    rcs_select::Dict   # Selection map thruster index <--> symbol

    """
        ApolloCSM()

    Constructor of the Apollo Command and Service Module.

    # Returns
    - `csm`: the command and service module object.
    """
    function ApolloCSM()::ApolloCSM
        H = homtransf
        cvu = convert_units

        # Dynamical frame with respect to structural frame
        H_SD = H(cvu.([933.9; 5.0; 4.7], :in, :m))

        # RCS frame with respect to structural frame
        ang_offset = 7+15/60
        H_SR = H(roll=-ang_offset)

        # Quad positions with respect to RCS frame
        Pan_RQ = Dict(:A=>H(cvu.([958.97; 0.0; -83.56], :in, :m)),
                      :B=>H(cvu.([958.97; 83.56; 0.0], :in, :m)),
                      :C=>H(cvu.([958.97; 0.0; 83.56], :in, :m)),
                      :D=>H(cvu.([958.97; -83.56; 0.0], :in, :m)))

        # Quad rotations with respect to RCS frame
        Rot_RQ = Dict(:A=>H(roll=-90),
                      :B=>H(roll=0),
                      :C=>H(roll=90),
                      :D=>H(roll=180))

        # Quad frames with respect to RCS frame
        H_RQ = Dict(k=>Pan_RQ[k]*Rot_RQ[k] for k in (:A, :B, :C, :D))

        # Thruster positions in quad frame
        Pan_QT = Dict(:pf=>H(cvu.([6.75, 0.0, 0.0], :in, :m)),
                      :pa=>H(cvu.([-6.75, 0.0, 0.0], :in, :m)),
                      :rf=>H(cvu.([0.94, 0.0, 3.125], :in, :m)),
                      :ra=>H(cvu.([-0.94, 0.0, -3.125], :in, :m)))

        # Thruster orientations with respect to quad frame
        cant = 10
        Rot_QT = Dict(:pf=>H(yaw=cant),
                      :pa=>H(pitch=180)*H(yaw=cant),
                      :rf=>H(pitch=-90)*H(yaw=cant),
                      :ra=>H(pitch=90)*H(yaw=cant))

        # Thruster frames with respect to quad frame
        H_QT = Dict(k=>Pan_QT[k]*Rot_QT[k] for k in (:pf, :pa, :rf, :ra))

        # Thrusters with respect to dynamical frame
        H_DT = Dict()
        Disp_DDA = hominv(H_SD)
        for quad in (:A, :B, :C, :D)
            for thruster in (:pf, :pa, :rf, :ra)
                k = (quad, thruster)
                H_DT[k] = Disp_DDA*H_SR*H_RQ[quad]*H_QT[thruster]
            end
        end

        # Thruster positions in dynamics frame
        r_rcs = Dict(k=>homdisp(H_DT[k]) for k in keys(H_DT))
        f_rcs_T = [-1; 0; 0] # Thrust vector in thruster frame
        f_rcs = Dict(k=>homrot(H_DT[k])*f_rcs_T for k in keys(H_DT))

        # Docking port position in structural frame
        H_SL = H(cvu.([1110.25; 0.0; 0.0], :in, :m))

        # Docking port in dynamical frame
        H_DL = Disp_DDA*H_SL

        # Mass properties
        m = convert_units(66850.6, :lb, :kg)
        J_xx, J_yy, J_zz = 36324, 80036, 81701
        J_xy, J_xz, J_yz = -2111, 273, 2268
        J = [J_xx -J_xy -J_xz;
             -J_xy J_yy -J_yz;
             -J_xz -J_yz J_zz]
        J = convert_units.(J, :ft2slug, :m2kg)

        # Propulsion properties
        imp_min = 50.0
        imp_max = 445.0

        # Thruster selection map converting back and forth between thruster
        # number and humand-readable thruster identifier using the quad and
        # thruster name
        rcs_select = Dict()
        thruster_count = 0
        for quad in (:A, :B, :C, :D)
            for thruster in (:pf, :pa, :rf, :ra)
                thruster_count += 1
                key = (quad, thruster)
                rcs_select[key] = thruster_count
                rcs_select[thruster_count] = key
            end
        end

        # Compile all into CSM object
        csm = new(H_SD, H_SR, H_RQ, H_QT, H_DT, H_DL, r_rcs, f_rcs, m, J,
                  imp_min, imp_max, rcs_select)

        return csm
    end # function
end # struct

""" Chaser vehicle parameters. """
struct ChaserParameters
    # Indices
    id_r::IntRange        # Inertial position (state)
    id_v::IntRange        # Inertial velocity (state)
    id_q::IntRange        # Quaternion (state)
    id_ω::IntRange        # Body frame angular velocity (state)
    id_rcs::IntRange      # RCS impulses (input)
    id_rcs_ref::IntRange  # RCS impulse references (input)
    id_rcs_eq::Int        # RCS impulse actual minus reference (input)
    id_t::Int             # Time dilation (parameter)
    id_dock_tol::IntRange # Docking tolerance (parameter)
    # Vehicle
    csm::ApolloCSM   # Apollo command and service module vehicle
end # struct

""" Planar rendezvous flight environment. """
struct RendezvousEnvironmentParameters
    xi::RealVector   # Inertial axis "normal out of dock port"
    yi::RealVector   # Inertial axis "dock port left (when facing)"
    zi::RealVector   # Inertial axis "dock port down (when facing)"
    n::RealValue     # [rad/s] Orbital mean motion
end

""" Parameters of the chaser trajectory. """
mutable struct RendezvousTrajectoryParameters
    # Boundary conditions
    r0::RealVector      # Initial position
    rf::RealVector      # Terminal position
    v0::RealVector      # Initial velocity
    vf::RealVector      # Terminal velocity
    q0::Quaternion      # Initial attitude
    qf::Quaternion      # Terminal attitude
    ω0::RealVector      # Initial angular velocity
    ωf::RealVector      # Terminal angular velocity
    # Docking tolerances
    v_axial_min::RealValue  # Minimum axial (closing) velocity
    v_axial_max::RealValue  # Maximum axial (closing) velocity
    v_radial_tol::RealValue # Radial (transverse) velocity
    r_radial_tol::RealValue # Radial alignment of x-axes
    ω_tol::RealValue        # Angular velocity (about any axis)
    x_ang_tol::RealValue    # Angular alignment of x-axes
    roll_tol::RealValue     # Roll attitude (about x axis)
    # Time of flight
    tf_min::RealValue   # Minimum flight time
    tf_max::RealValue   # Maximum flight time
    # Homotopy
    κ1::RealValue       # Sigmoid homotopy parameter
    κ1_grid::RealVector # Sweep of all homotopy parameters
    κ2::RealValue       # Normalization homotopy parameter
    β::RealValue        # Relative cost improvement triggering homotopy update
    γ::RealValue        # Control weight for deadband relaxation
end # struct

""" Rendezvous trajectory optimization problem parameters all in one. """
struct RendezvousProblem
    vehicle::ChaserParameters            # The ego-vehicle
    env::RendezvousEnvironmentParameters # The environment
    traj::RendezvousTrajectoryParameters # The trajectory
end

# ..:: Methods ::..

"""
    RendezvousProblem()

Constructor for the "full" 3D rendezvous problem.

# Returns
- `mdl`: the problem definition object.
"""
function RendezvousProblem()::RendezvousProblem

    # ..:: Environment ::..
    xi = [1.0; 0.0; 0.0]
    yi = [0.0; 1.0; 0.0]
    zi = [0.0; 0.0; 1.0]
    μ = 3.986e14 # [m³/s²] Standard gravitational parameter
    Re = 6378e3 # [m] Earth radius
    R = Re+400e3 # [m] Orbit radius
    n = sqrt(μ/R^3)
    env = RendezvousEnvironmentParameters(xi, yi, zi, n)

    # ..:: Chaser spacecraft ::..
    # Indices
    id_r = 1:3
    id_v = 4:6
    id_q = 7:10
    id_ω = 11:13
    id_rcs = 1:16
    id_rcs_ref = (1:16).+id_rcs[end]
    id_rcs_eq = id_rcs_ref[end]+1
    id_t = 1
    id_dock_tol = (1:13).+1
    # Vehicle
    csm = ApolloCSM()

    sc = ChaserParameters(id_r, id_v, id_q, id_ω, id_rcs, id_rcs_ref,
                          id_rcs_eq, id_t, id_dock_tol, csm)

    # ..:: Trajectory ::..
    # >> Boundary conditions <<
    r0 = 50.0*xi+10.0*zi+5.0*yi
    rf = 0.0*xi
    v0 = 0.0*xi
    vf = -0.1*xi
    ω0 = zeros(3)
    ωf = zeros(3)
    # >> Docking tolerances <<
    v_axial_min = convert_units(0.1, :ftps, :mps)
    v_axial_max = convert_units(1.0, :ftps, :mps)
    v_radial_tol = convert_units(0.5, :ftps, :mps)
    r_radial_tol = convert_units(1.0, :ft, :m)
    ω_tol = convert_units(0.1, :deg, :rad)
    x_ang_tol = convert_units(10.0, :deg, :rad)
    roll_tol = convert_units(10.0, :deg, :rad)
    # Docking port (inertial) frame
    # Baseline docked configuration
    q_dock = Quaternion(deg2rad(180), yi)*Quaternion(deg2rad(180), xi)
    q_init = q_dock*Quaternion(deg2rad(180), zi)*Quaternion(deg2rad(10), yi)
    q0 = q_init
    qf = q_dock
    # >> Time of flight <<
    tf_min = 100.0
    tf_max = 500.0
    # >> Homotopy <<
    κ2 = 1.0
    β = 10e0/100
    γ = 3e-1
    Nhom = 10 # Number of homotopy values to sweep through
    hom_grid = LinRange(0.0, 1.0, Nhom)
    hom_κ1 = Homotopy(1e-4; δ_max=5.0) #noerr
    κ1_grid = [hom_κ1(v) for v in hom_grid]
    κ1 = κ1_grid[1]

    traj = RendezvousTrajectoryParameters(
        r0, rf, v0, vf, q0, qf, ω0, ωf,
        v_axial_min, v_axial_max, v_radial_tol, r_radial_tol, ω_tol,
        x_ang_tol, roll_tol, tf_min, tf_max, κ1, κ1_grid, κ2, β, γ)

    mdl = RendezvousProblem(sc, env, traj)

    return mdl
end # function
