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

using ..SCPToolbox

# ..:: Globals ::..

const RCSKey = Tuple{Symbol,Symbol}

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
    H_RQ::Dict{Symbol,RealMatrix}
    H_QT::Dict{Symbol,RealMatrix}
    H_DT::Dict{RCSKey,RealMatrix}
    H_DP::RealMatrix
    # Vectors (in D frame)
    r_rcs::Dict{RCSKey,RealVector} # Thrust application points
    f_rcs::Dict{RCSKey,RealVector} # Thrust vectors
    # Mass properties
    m::RealValue       # [kg] Mass
    J::RealMatrix      # [kg] Inertia matrix in D frame
    # Propulsion properties
    imp_min::RealValue # [N*s] Minimum RCS thruster impulse
    imp_max::RealValue # [N*s] Maximum RCS thruster impulse
    Frcs::RealValue    # [N] Constant thrust produced when firing
    fuel::Function     # [kg/s] Fuel consumption rate for one thruster
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
        ang_offset = 7 + 15 / 60
        H_SR = H(roll = -ang_offset)

        # Quad positions with respect to RCS frame
        Pan_RQ = Dict(
            :A => H(cvu.([958.97; 0.0; -83.56], :in, :m)),
            :B => H(cvu.([958.97; 83.56; 0.0], :in, :m)),
            :C => H(cvu.([958.97; 0.0; 83.56], :in, :m)),
            :D => H(cvu.([958.97; -83.56; 0.0], :in, :m)),
        )

        # Quad rotations with respect to RCS frame
        Rot_RQ = Dict(
            :A => H(roll = -90),
            :B => H(roll = 0),
            :C => H(roll = 90),
            :D => H(roll = 180),
        )

        # Quad frames with respect to RCS frame
        H_RQ = Dict(k => Pan_RQ[k] * Rot_RQ[k] for k in (:A, :B, :C, :D))

        # Thruster positions in quad frame
        Pan_QT = Dict(
            :pf => H(cvu.([6.75, 0.0, 0.0], :in, :m)),
            :pa => H(cvu.([-6.75, 0.0, 0.0], :in, :m)),
            :rf => H(cvu.([0.94, 0.0, 3.125], :in, :m)),
            :ra => H(cvu.([-0.94, 0.0, -3.125], :in, :m)),
        )

        # Thruster orientations with respect to quad frame
        cant = 10
        Rot_QT = Dict(
            :pf => H(yaw = cant),
            :pa => H(pitch = 180) * H(yaw = cant),
            :rf => H(pitch = -90) * H(yaw = cant),
            :ra => H(pitch = 90) * H(yaw = cant),
        )

        # Thruster frames with respect to quad frame
        H_QT = Dict(k => Pan_QT[k] * Rot_QT[k] for k in (:pf, :pa, :rf, :ra))

        # Thrusters with respect to dynamical frame
        H_DT = Dict()
        H_DS = hominv(H_SD)
        for quad in (:A, :B, :C, :D)
            for thruster in (:pf, :pa, :rf, :ra)
                k = (quad, thruster)
                H_DT[k] = H_DS * H_SR * H_RQ[quad] * H_QT[thruster]
            end
        end

        # Thruster positions in dynamics frame
        r_rcs = Dict(k => homdisp(H_DT[k]) for k in keys(H_DT))
        f_rcs_T = [-1; 0; 0] # Thrust vector in thruster frame
        f_rcs = Dict(k => homrot(H_DT[k]) * f_rcs_T for k in keys(H_DT))

        # Docking port position in structural frame
        docked_orientation = -30
        H_SP = H(cvu.([1110.25; 0.0; 0.0], :in, :m))
        H_SP *= H(roll = docked_orientation)

        # Docking port in dynamical frame
        H_DP = H_DS * H_SP

        # Mass properties
        m = convert_units(66850.6, :lb, :kg)
        J_xx, J_yy, J_zz = 36324, 80036, 81701
        J_xy, J_xz, J_yz = -2111, 273, 2268
        J = [
            J_xx -J_xy -J_xz
            -J_xy J_yy -J_yz
            -J_xz -J_yz J_zz
        ]
        J = convert_units.(J, :ft2slug, :m2kg)

        # Propulsion properties
        imp_min = 50.0
        imp_max = 445.0
        Frcs = 445.0

        # Piecewise affine map for single thruster fuel consumption map (pulse
        # duration to fuel consumed)
        pulse = [
            36.552
            50.042
            63.532
            77.022
            90.512
            104.002
            117.492
            130.982
            144.472
            157.962
            171.452
            184.942
        ]
        pulse = (pulse .- pulse[1]) ./ (pulse[end] - pulse[1]) .* (1000 - 14) .+ 14
        pulse .*= 1e-3
        pushfirst!(pulse, 0)

        fuel = [
            108.433
            108.121
            107.656
            106.527
            105.143
            103.189
            99.941
            94.992
            87.920
            78.050
            62.829
            40.849
        ]
        fuel = (fuel .- fuel[1]) ./ (fuel[end] - fuel[1]) .* (0.364 - 0.005) .+ 0.005
        fuel = convert_units.(fuel, :lb, :kg)
        pushfirst!(fuel, 0)

        fuel_consum = t -> linterp(t, fuel, pulse)

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
        csm = new(
            H_SD,
            H_SR,
            H_RQ,
            H_QT,
            H_DT,
            H_DP,
            r_rcs,
            f_rcs,
            m,
            J,
            imp_min,
            imp_max,
            Frcs,
            fuel_consum,
            rcs_select,
        )

        return csm
    end
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
    r0::RealVector       # Initial position
    rf::RealVector       # Terminal position
    v0::RealVector       # Initial velocity
    vf::RealVector       # Terminal velocity
    q0::Quaternion       # Initial attitude
    qf::Quaternion       # Terminal attitude
    ω0::RealVector       # Initial angular velocity
    ωf::RealVector       # Terminal angular velocity
    # Docking tolerances
    rf_tol::RealValue    # Radial x-axis alignment
    vf_tol::RealValue    # Velocity mismatch (in all axes)
    ang_tol::RealValue   # Angular mismatch (net angle)
    ωf_tol::RealValue    # Angular velocity mismatch (in all axes)
    # Plume impingement
    r_plume::RealValue   # Approach radius below which plume impingement danger
    # Maneuver approach cone
    r_appch::RealValue   # Approach sphere radius
    θ_appch::RealValue   # Approach cone half-angle
    # Time of flight
    tf_min::RealValue    # Minimum flight time
    tf_max::RealValue    # Maximum flight time
    # Homotopy
    hom::RealValue       # Sigmoid homotopy parameter
    hom_grid::RealVector # Sweep of all homotopy parameters
    β::RealValue         # Relative cost improvement triggering homotopy update
    γc::RealValue        # Deadband relaxation cost weight
    γg::RealValue        # Deadband relaxation gradient keepout zone
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
    R = Re + 400e3 # [m] Orbit radius
    n = sqrt(μ / R^3)
    env = RendezvousEnvironmentParameters(xi, yi, zi, n)

    # ..:: Chaser spacecraft ::..
    # Indices
    id_r = 1:3
    id_v = 4:6
    id_q = 7:10
    id_ω = 11:13
    id_rcs = 1:16
    id_rcs_ref = (1:16) .+ id_rcs[end]
    id_rcs_eq = id_rcs_ref[end] + 1
    id_t = 1
    id_dock_tol = (1:13) .+ 1
    # Vehicle
    csm = ApolloCSM()

    sc = ChaserParameters(
        id_r,
        id_v,
        id_q,
        id_ω,
        id_rcs,
        id_rcs_ref,
        id_rcs_eq,
        id_t,
        id_dock_tol,
        csm,
    )

    # ..:: Trajectory ::..
    # >> Boundary conditions <<
    r0 = 100.0 * xi - 20.0 * zi + 20.0 * yi
    v0 = 0.0 * xi
    vf = -0.1 * xi
    q0 = Quaternion(deg2rad(0), yi)
    ω0 = zeros(3)
    ωf = zeros(3)
    # Terminal pose based on docking orientation
    H_LP = homtransf(yaw = 180)
    H_LD = H_LP * hominv(csm.H_DP)
    rf = homdisp(H_LD)
    Rf = homrot(H_LD)
    qf = Quaternion(Rf)
    # >> Docking tolerances <<
    rf_tol = 0.1
    vf_tol = 0.01
    ang_tol = deg2rad(1)
    ωf_tol = deg2rad(0.01)
    # >> Plume and approach spheres <<
    r_plume = 20
    r_appch = 30
    θ_appch = deg2rad(10)
    # >> Time of flight <<
    tf_min = 100.0
    tf_max = 1000.0
    # >> Homotopy <<
    β = 1e1 / 100
    γc = 1.0
    γg = 5.0
    hom_steps = 10 # Number of homotopy values to sweep through
    hom_obj = Homotopy(1e-2; δ_max = 10.0)
    hom_grid = map(hom_obj, LinRange(0.0, 1.0, hom_steps))
    hom = hom_grid[1]

    traj = RendezvousTrajectoryParameters(
        r0,
        rf,
        v0,
        vf,
        q0,
        qf,
        ω0,
        ωf,
        rf_tol,
        vf_tol,
        ang_tol,
        ωf_tol,
        r_plume,
        r_appch,
        θ_appch,
        tf_min,
        tf_max,
        hom,
        hom_grid,
        β,
        γc,
        γg,
    )

    mdl = RendezvousProblem(sc, env, traj)

    return mdl
end

"""
    fuel_consumption(mdl, impulses)

Compute the Apollo CSM fuel consumption given a history of thruster impulses.

# Arguments
- `mdl`: the problem definition object.
- `impulses`: a matrix of impulses where each column stores the duration of
  impulses of each thruster for that control opportunity.

# Returns
- `fuel`: the amount of fuel consumed by this impulse history.
"""
function fuel_consumption(mdl::RendezvousProblem, impulses::RealMatrix)::RealValue

    # Extract pulse durations
    csm = mdl.vehicle.csm
    dt = impulses ./ csm.Frcs
    N = size(dt, 2) # History length

    # Integrate the fuel consumption
    fuel = 0.0
    for k = 1:N
        dtk = dt[:, k]
        fuel_k = sum(csm.fuel.(dtk))
        fuel += fuel_k
        # dtk = sort(dtk)
        # diff_dtk = diff(dtk)
        # pushfirst!(diff_dtk, dtk[1])
        # csm.c1*sum((nrcs-i+1)^2*diff_dtk[i] for i=1:nrcs)
    end

    return fuel
end
