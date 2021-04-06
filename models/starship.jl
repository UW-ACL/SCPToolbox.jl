#= Starship landing flip maneuver data structures and custom methods.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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

using PyPlot
using Colors

include("../utils/types.jl")
include("../core/problem.jl")
include("../core/scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Starship vehicle parameters. =#
struct StarshipParameters
    # ..:: Indices ::..
    id_r::T_IntRange   # Position indices of the state vector
    id_v::T_IntRange   # Velocity indices of the state vector
    id_θ::T_Int        # Tilt angle index of the state vector
    id_ω::T_Int        # Tilt rate index of the state vector
    id_m::T_Int        # Mass index of the state vector
    id_δd::T_Int       # Delayed gimbal angle index of the state vector
    id_τ::T_Int        # Normalized time index of the state vector
    id_T::T_Int        # Thrust index of the input vector
    id_δ::T_Int        # Gimbal angle index of the input vector
    id_δdot::T_Int     # Gimbal rate index of the input vector
    id_t1::T_Int       # First phase duration index of parameter vector
    id_t2::T_Int       # Second phase duration index of parameter vector
    # ..:: Body axes ::..
    ei::T_Function     # Lateral body axis in body or world frame
    ej::T_Function     # Longitudinal body axis in body or world frame
    # ..:: Mechanical parameters ::..
    lcg::T_Real        # [m] CG location (from base)
    lcp::T_Real        # [m] CP location (from base)
    m::T_Real          # [kg] Total mass
    J::T_Real          # [kg*m^2] Moment of inertia about CG
    # ..:: Aerodynamic parameters ::..
    CD::T_Real         # [kg/m] Overall drag coefficient 0.5*ρ*cd*A
    # ..:: Propulsion parameters ::..
    T_min1::T_Real     # [N] Minimum thrust (one engine)
    T_max1::T_Real     # [N] Maximum thrust (one engine)
    T_min3::T_Real     # [N] Minimum thrust (three engines)
    T_max3::T_Real     # [N] Maximum thrust (three engines)
    αe::T_Real         # [s/m] Mass depletion propotionality constant
    δ_max::T_Real      # [rad] Maximum gimbal angle
    δdot_max::T_Real   # [rad/s] Maximum gimbal rate
    rate_delay::T_Real # [s] Delay for approximate rate constraint
end

#= Starship flight environment. =#
struct StarshipEnvironmentParameters
    ex::T_RealVector # Horizontal "along" axis
    ey::T_RealVector # Vertical "up" axis
    g::T_RealVector  # [m/s^2] Gravity vector
end

#= Trajectory parameters. =#
struct StarshipTrajectoryParameters
    r0::T_RealVector # [m] Initial position
    v0::T_RealVector # [m/s] Initial velocity
    θ0::T_Real       # [rad] Initial tilt angle
    vf::T_RealVector # [m/s] Terminal velocity
    θf::T_Real       # [rad] Terminal tilt angle
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
    γ_gs::T_Real     # [rad] Maximum glideslope (measured from vertical)
    τs::T_Real       # Normalized time end of first phase
end

#= Starship trajectory optimization problem parameters all in one. =#
struct StarshipProblem
    vehicle::StarshipParameters        # The ego-vehicle
    env::StarshipEnvironmentParameters # The environment
    traj::StarshipTrajectoryParameters # The trajectory
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Constructor for the Starship landing flip maneuver problem.

Returns:
    mdl: the problem definition object. =#
function StarshipProblem()::StarshipProblem

    # ..:: Environment ::..
    ex = [1.0; 0.0]
    ey = [0.0; 1.0]
    g0 = 9.81 # [m/s^2] Gravitational acceleration
    g = -g0*ey
    env = StarshipEnvironmentParameters(ex, ey, g)

    # ..:: Starship ::..
    # >> Indices <<
    id_r = 1:2
    id_v = 3:4
    id_θ = 5
    id_ω = 6
    id_m = 7
    id_δd = 8
    id_τ = 9
    id_T = 1
    id_δ = 2
    id_δdot = 3
    id_t1 = 1
    id_t2 = 2
    # >> Body axes <<
    ei = (θ) -> cos(θ)*[1.0; 0.0]+sin(θ)*[0.0; 1.0]
    ej = (θ) -> -sin(θ)*[1.0; 0.0]+cos(θ)*[0.0; 1.0]
    # >> Mechanical parameters <<
    rs = 4.5 # [m] Fuselage radius
    ls = 50.0 # [m] Fuselage height
    m = 120e3
    lcg = 0.5*ls
    lcp = 0.4*ls
    J = 1/12*m*(6*rs^2+ls^2)
    # >> Aerodynamic parameters <<
    vterm = 75 # [m/s] Terminal velocity (during freefall)
    CD = m*g0/vterm^2
    CD *= 1.2 # Fudge factor
    # >> Propulsion parameters <<
    Isp = 330 # [s] Specific impulse
    T_min1 = 880e3 # [N] One engine min thrust
    T_max1 = 2210e3 # [N] One engine max thrust
    T_min3 = 3*T_min1
    T_max3 = 3*T_max1
    αe = -1/(Isp*g0)
    δ_max = deg2rad(10.0)
    δdot_max = 2*δ_max
    rate_delay = 0.05

    starship = StarshipParameters(
        id_r, id_v, id_θ, id_ω, id_m, id_δd, id_τ, id_T, id_δ, id_δdot,
        id_t1, id_t2, ei, ej, lcg, lcp, m, J, CD, T_min1, T_max1, T_min3,
        T_max3, αe, δ_max, δdot_max, rate_delay)

    # ..:: Trajectory ::..
    r0 = 100.0*ex+550.0*ey
    v0 = -vterm*ey
    vf = -5.0*ey
    θ0 = deg2rad(90.0)
    θf = deg2rad(-10.0)
    tf_min = 0.0
    tf_max = 30.0
    γ_gs = deg2rad(27.0)
    τs = 0.5
    traj = StarshipTrajectoryParameters(r0, v0, θ0, vf, θf, tf_min,
                                        tf_max, γ_gs, τs)

    mdl = StarshipProblem(starship, env, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
Starship vehicle dynamics.

Args:
* `x`: the current state vector.
* `u`: the current input vector.
* `p`: the parameter vector.
* `pbm`: the Starship landing flip problem description.
* `no_aero_torques`: (optional) whether to omit torques generated by lift and
  drag.

Returns:
* `f`: the time derivative of the state vector.
"""
function dynamics(x::T_RealVector,
                  u::T_RealVector,
                  p::T_RealVector,
                  pbm::TrajectoryProblem;
                  no_aero_torques::T_Bool=false)::T_RealVector

    # Parameters
    veh = pbm.mdl.vehicle
    env = pbm.mdl.env
    traj = pbm.mdl.traj

    # Current (x, u, p) values
    v = x[veh.id_v]
    θ = x[veh.id_θ]
    ω = x[veh.id_ω]
    m = x[veh.id_m]
    τ = x[veh.id_τ]
    δd = x[veh.id_δd]
    T = u[veh.id_T]
    δ = u[veh.id_δ]
    tdil = ((τ<=traj.τs) ? p[veh.id_t1]/traj.τs :
            p[veh.id_t2]/(1-traj.τs))

    # Derived quantities
    ℓeng = -veh.lcg
    ℓcp = veh.lcp-veh.lcg
    ei = veh.ei(θ)
    ej = veh.ej(θ)
    Tv = T*(-sin(δ)*ei+cos(δ)*ej)
    MT = ℓeng*T*sin(δ)
    D = -veh.CD*norm(v)*v
    if !no_aero_torques
        MD = -ℓcp*dot(D, ei)
    else
        MD = 0.0
    end

    # The dynamics
    f = zeros(pbm.nx)
    f[veh.id_r] = v
    f[veh.id_v] = (Tv+D)/veh.m+env.g
    f[veh.id_θ] = ω
    f[veh.id_ω] = (MT+MD)/veh.J
    f[veh.id_m] = veh.αe*T
    f[veh.id_δd] = (δ-δd)/veh.rate_delay
    f[veh.id_τ] = 1.0

    # Scale for time
    f *= tdil
    f[veh.id_τ] /= tdil

    return f
end

#= Plot the final converged trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_final_trajectory(mdl::StarshipProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = get_colormap()(1.0)
    N = size(sol.xd, 2)
    speed = [norm(@k(sol.xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_cmap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)

    fig = create_figure((3, 4))
    ax = fig.add_subplot()

    ax.axis("equal")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("Downrange [m]")
    ax.set_ylabel("Altitude [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity [m/s]")

    # ..:: Draw the glide slope constraint ::..
    alt = 200.0 # [m] Altitude of glide slope "triangle" visualization
    x_gs = alt*tan(mdl.traj.γ_gs)
    ax.plot([-x_gs, 0, x_gs], [alt, 0, alt],
            color="#5da9a1",
            linestyle="--",
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=90)

    # ..:: Draw the final continuous-time position trajectory ::..
    # Collect the continuous-time trajectory data
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_pos = T_RealMatrix(undef, 2, ct_res)
    ct_speed = T_RealVector(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, @k(ct_τ))
        @k(ct_pos) = xk[mdl.vehicle.id_r[1:2]]
        @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res-1
        r, v = @k(ct_pos), @k(ct_speed)
        x, y = r[1], r[2]
        ax.plot(x, y,
                linestyle="none",
                marker="o",
                markersize=4,
                alpha=0.2,
                markerfacecolor=v_cmap.to_rgba(v),
                markeredgecolor="none",
                clip_on=false,
                zorder=100)
    end

    # ..:: Draw the acceleration vector ::..
    T = sol.ud[mdl.vehicle.id_T, :]
    θ = sol.xd[mdl.vehicle.id_θ, :]
    δ = sol.ud[mdl.vehicle.id_δ[1], :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    u_nrml = maximum(T)
    r_span = norm(mdl.traj.r0)
    u_scale = 1/u_nrml*r_span*0.1
    for k = 1:N
        base = pos[1:2, k]
        thrust = -[-T[k]*sin(θ[k]+δ[k]); T[k]*cos(θ[k]+δ[k])]
        tip = base+u_scale*thrust
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                zorder=100)
    end

    # ..:: Draw the fuselage ::..
    b_scale = r_span*0.1
    num_draw = 6 # Number of instances to draw
    K = T_IntVector(1:(N÷num_draw):N)
    for k = 1:N
        altitude = dot(@k(pos), mdl.env.ey)
        if altitude>=0 || k==N || k in K
            base = pos[1:2, k]
            nose = [-sin(θ[k]); cos(θ[k])]
            tip = base+b_scale*nose
            x = [base[1], tip[1]]
            y = [base[2], tip[2]]
            ax.plot(x, y,
                    color="#26415d",
                    linewidth=1.5,
                    solid_capstyle="round",
                    zorder=100)
        end
    end

    # ..:: Draw the discrete-time positions trajectory ::..
    pos = sol.xd[mdl.vehicle.id_r, :]
    x, y = pos[1, :], pos[2, :]
    ax.plot(x, y,
            linestyle="none",
            marker="o",
            markersize=3,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            clip_on=false,
            zorder=100)

    save_figure("starship_final_traj", algo)

    return nothing
end

#= Plot the thrust trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution. =#
function plot_thrust(mdl::StarshipProblem,
                     sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    traj = mdl.traj
    clr = get_colormap()(1.0)
    t1 = sol.p[veh.id_t1]
    t2 = sol.p[veh.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)
    N = size(sol.xd, 2)
    scale = 1e-6
    y_top = 7.0
    y_bot = 0.0

    fig = create_figure((5, 2.5))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [MN]")

    # ..:: Thrust bounds ::..
    bnd_min1 = veh.T_min1*scale
    bnd_max1 = veh.T_max1*scale
    bnd_min3 = veh.T_min3*scale
    bnd_max3 = veh.T_max3*scale
    plot_timeseries_bound!(ax, 0.0, t1, bnd_max3, y_top-bnd_max3)
    plot_timeseries_bound!(ax, 0.0, t1, bnd_min3, y_bot-bnd_min3)
    plot_timeseries_bound!(ax, t1, tf, bnd_max1, y_top-bnd_max1)
    plot_timeseries_bound!(ax, t1, tf, bnd_min1, y_bot-bnd_min1)

    # ..:: Thrust value (continuous-time) ::..
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    ct_thrust = T_RealVector([sample(sol.uc, τ)[mdl.vehicle.id_T]*scale
                              for τ in ct_τ])
    ax.plot(ct_time, ct_thrust,
            color=clr,
            linewidth=2)

    # ..:: Thrust value (discrete-time) ::..
    dt_time = map(τ2t, sol.τd)
    dt_thrust = sol.ud[mdl.vehicle.id_T, :]*scale
    ax.plot(dt_time, dt_thrust,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    save_figure("starship_thrust", algo)

    return nothing
end

#= Plot the gimbal angle trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution. =#
function plot_gimbal(mdl::StarshipProblem,
                     sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    clr = get_colormap()(1.0)
    t1 = sol.p[mdl.vehicle.id_t1]
    t2 = sol.p[mdl.vehicle.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    scale = 180/pi
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)

    fig = create_figure((5, 5))

    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    dt_time = map(τ2t, sol.τd)

    # ..:: Gimbal angle timeseries ::..
    ax = fig.add_subplot(211)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gimbal angle [\$^\\circ\$]")

    # >> Gimbal angle bounds <<
    pad = 2.0
    bnd_max = mdl.vehicle.δ_max*scale
    bnd_min = -mdl.vehicle.δ_max*scale
    y_top = bnd_max+pad
    y_bot = bnd_min-pad
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # >> Delayed gimbal angle (continuous-time) <<
    ct_gimbal_delayed = T_RealVector([
        sample(sol.xc, τ)[mdl.vehicle.id_δd]*scale for τ in ct_τ])
    ax.plot(ct_time, ct_gimbal_delayed,
            color="#db6245",
            linestyle="--",
            linewidth=1,
            dash_capstyle="round")

    # >> Delayed gimbal angle (discrete-time) <<
    dt_gimbal_delayed = sol.xd[mdl.vehicle.id_δd, :]*scale
    ax.plot(dt_time, dt_gimbal_delayed,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0,
            markerfacecolor="#db6245",
            clip_on=false)

    # >> Gimbal angle (continuous-time) <<
    ct_gimbal = T_RealVector([
        sample(sol.uc, τ)[mdl.vehicle.id_δ]*scale for τ in ct_τ])
    ax.plot(ct_time, ct_gimbal,
            color=clr,
            linewidth=2)

    # >> Gimbal angle (discrete-time) <<
    dt_gimbal = sol.ud[mdl.vehicle.id_δ, :]*scale
    ax.plot(dt_time, dt_gimbal,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # ..:: Gimbal rate timeseries ::..
    ax = fig.add_subplot(212)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gimbal rate [\$^\\circ\$/s]")

    # >> Gimbal rate bounds <<
    pad = 2.0
    bnd_max = mdl.vehicle.δdot_max*scale
    bnd_min = -mdl.vehicle.δdot_max*scale
    y_top = bnd_max+pad
    y_bot = bnd_min-pad
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # >> Actual gimbal rate (discrete-time) <<
    δ = sol.ud[mdl.vehicle.id_δ, 1:end-1]
    δn = sol.ud[mdl.vehicle.id_δ, 2:end]
    dt_δdot = (δn-δ)./diff(dt_time)
    ax.plot(dt_time[2:end], dt_δdot*scale,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # >> Actual gimbal rate (continuous-time) <<
    δdot = T_ContinuousTimeTrajectory(sol.τd[1:end-1], dt_δdot, :zoh)
    ct_δdot = T_RealVector([sample(δdot, τ)*scale for τ in ct_τ])
    ax.plot(ct_time, ct_δdot,
            color=clr,
            linewidth=2,
            zorder=100)

    # >> Constraint (approximate) gimbal rate (discrete-time) <<
    δ = sol.xd[mdl.vehicle.id_δd, :]
    δn = sol.ud[mdl.vehicle.id_δ, :]
    dt_δdot = (δn-δ)./mdl.vehicle.rate_delay
    ax.plot(dt_time, dt_δdot*scale,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0,
            markerfacecolor="#db6245",
            clip_on=false,
            zorder=110)

    save_figure("starship_gimbal", algo)

    return nothing
end
