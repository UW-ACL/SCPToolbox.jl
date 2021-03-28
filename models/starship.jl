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
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_θ::T_Int      # Tilt angle index of the state vector
    id_ω::T_Int      # Tilt rate index of the state vector
    id_m::T_Int      # Mass index of the state vector
    id_γ::T_Int      # Delayed gimbal angle index of the state vector
    id_ψ::T_Int      # Delayed fin area index of the state vector
    id_T::T_Int      # Thrust index of the input vector
    id_δ::T_Int      # Gimbal angle index of the input vector
    id_φ::T_Int      # Fin control index of the input vector
    id_t::T_Int      # Index of time dilation
    # ..:: Body axes ::..
    ei::T_RealVector # Lateral body axis
    ej::T_RealVector # Longitudinal body axis
    # ..:: Mechanical parameters ::..
    ls::T_Real       # [m] Total height
    le::T_Real       # [m] Length base to engine web
    lCH4::T_Real     # [m] Length base to CH4 header tank
    lO2::T_Real      # [m] Length base to O2 header tank
    lf::T_Real       # [m] Length base to fin center of pressure
    lcp::T_Real      # [m] Length base to fuselage center of pressure
    me::T_Real       # [kg] Engine web mass
    ms::T_Real       # [kg] Structure dry mass
    mO2::T_Real      # [kg] Initial O2 header tank total weight
    mCH4::T_Real     # [kg] Initial CH4 header tank total weight
    mwet::T_Real     # [kg] Vehicle wet mass
    Js0::T_Real      # [kg*m^2] Fuselage moment of inertia about center
    Afin::T_Real     # [m^2] Maximum wind-facing front fin area
    # ..:: Aerodynamic parameters ::..
    CDfin::T_Real    # [kg/m^2] 0.5*ρ*CD for one fin
    CDs0::T_Real     # [kg/m^2] 0.5*ρ*CD*A for fuselage cylinder end-on
    CDs1::T_Real     # [kg/m^2] 0.5*ρ*CD*A for fuselage cylinder front-on
    vterm::T_Real    # [m/s] Terminal velocity (during freefall)
    # ..:: Propulsion parameters ::..
    σ::T_Real        # [-] Mass combustion ratio
    T_max::T_Real    # [N] Maximum thrust
    T_min::T_Real    # [N] Minimum thrust
    αe::T_Real       # [s/m] Mass depletion propotionality constant
    δ_max::T_Real    # [rad] Maximum gimbal angle
    β_max::T_Real    # [rad/s] Maximum gimbal rate
    α_max::T_Real    # [m^2/s] Maximum fin area rate
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
    vf::T_RealVector # [m/s] Terminal velocity
    θ0::T_Real       # [rad] Initial tilt angle
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
    γ_gs::T_Real     # [rad] Maximum glideslope (measured from vertical)
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

    # ..:: Starship ::..
    # >> Indices <<
    id_r = 1:2
    id_v = 3:4
    id_θ = 5
    id_ω = 6
    id_m = 7
    id_γ = 8
    id_ψ = 9
    id_T = 1
    id_δ = 2
    id_φ = 3
    id_t = 1
    # >> Body axes <<
    ei = [1.0; 0.0]
    ej = [0.0; 1.0]
    # >> Mechanical parameters <<
    rs = 4.5 # [m] Fuselage radius
    ls = 50.0
    le = 3.19
    lCH4 = 16.05
    lO2 = 48.05
    lf = 25.21
    lcp = ls*0.5
    me = 25e3
    ms = 65e3
    mO2s = 600 # [kg] O2 header tank structure weight
    mCH4s = 600 # [kg] CH4 header tank structure weight
    VO2 = 18.67 # [m^3] O2 header tank volume
    VCH4 = 16.21 # [m^3] CH4 header tank volume
    ρO2 = 1230 # [kg/m^3] Liquid O2 density
    ρCH4 = 451 # [kg/m^3] Liquid CH4 density
    mO2 = ρO2*VO2+mO2s
    mCH4 = ρCH4*VCH4+mCH4s
    mwet = me+ms+mO2+mCH4
    Js0 = 1/12*ms*(6*rs^2+ls^2)
    Afin = 27.0
    # >> Aerodynamic parameters <<
    g0 = 9.81 # [m/s^2] Gravitational acceleration
    ρa = 1.225 # [kg/m^3] Air density
    Sref0 = pi*rs^2 # [m^2] End-on cylinder fuselage area
    Sref1 = 2*rs*ls # [m^2] Front-on cylinder fuselage area
    vterm = 75
    _CDfin = 1.28
    _CDs0 = 0.8
    _CDs1 = 2*mwet*g0/(ρa*Sref1*vterm^2)
    CDfin = 0.5*ρa*_CDfin
    CDs0 = 0.5*ρa*_CDs0*Sref0
    CDs1 = 0.5*ρa*_CDs1*Sref1
    # >> Propulsion parameters <<
    ne = 3 # Number of engines
    g0 = 9.81 # [m/s^2] Acceleration due to gravity at sea level
    Isp = 330 # [s] Specific impulse
    σ_vol = 3.8 # [-] Volume cobustion ratio O2:CH4
    σ = ρO2*σ_vol/ρCH4
    T_min1 = 880e3 # [N] One engine min thrust
    T_max1 = 2210e3 # [N] One engine max thrust
    T_max = ne*T_max1
    T_min = T_min1
    αe = 1/((1+σ)*Isp*g0)
    δ_max = deg2rad(8.0)
    β_max = δ_max
    α_max = Afin/2
    rate_delay = 0.1

    starship = StarshipParameters(
        id_r, id_v, id_θ, id_ω, id_m, id_γ, id_ψ, id_T, id_δ, id_φ, id_t, ei,
        ej, ls, le, lCH4, lO2, lf, lcp, me, ms, mO2, mCH4, mwet, Js0, Afin,
        CDfin, CDs0, CDs1, vterm, σ, T_max, T_min, αe, δ_max, β_max, α_max,
        rate_delay)

    # ..:: Environment ::..
    ex = [1.0; 0.0]
    ey = [0.0; 1.0]
    g = -g0*ey
    env = StarshipEnvironmentParameters(ex, ey, g)

    # ..:: Trajectory ::..
    r0 = 100.0*ex+600.0*ey
    v0 = -vterm*ey
    vf = 0.0*ey
    θ0 = deg2rad(90.0)
    tf_min = 0.0
    tf_max = 60.0
    γ_gs = deg2rad(27.0)
    traj = StarshipTrajectoryParameters(r0, v0, vf, θ0, tf_min, tf_max, γ_gs)

    mdl = StarshipProblem(starship, env, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the initial discrete-time trajectory guess.

Use straight-line interpolation and a thrust that opposes gravity ("hover").

Args:
    pbm: the trajectory problem definition. =#
function starship_set_initial_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(pbm, (N, pbm) -> begin
                       veh = pbm.mdl.vehicle
                       traj = pbm.mdl.traj
                       env = pbm.mdl.env

                       # Parameter guess
                       p = zeros(pbm.np)
                       p[veh.id_t] = 0.5*(traj.tf_min+traj.tf_max)

                       # State guess
                       v_cst = -traj.r0/p[veh.id_t]
                       ω_cst = -traj.θ0/p[veh.id_t]
                       T_cst = norm(veh.mwet*env.g) # [N] Hover thrust
                       fuel_consum = p[veh.id_t]*veh.αe*T_cst
                       x0 = zeros(pbm.nx)
                       xf = zeros(pbm.nx)
                       x0[veh.id_r] = traj.r0
                       xf[veh.id_r] = zeros(2)
                       x0[veh.id_v] = v_cst
                       xf[veh.id_v] = v_cst
                       x0[veh.id_θ] = traj.θ0
                       xf[veh.id_θ] = 0.0
                       x0[veh.id_ω] = ω_cst
                       xf[veh.id_ω] = ω_cst
                       x0[veh.id_m] = 0.0
                       xf[veh.id_m] = fuel_consum
                       x0[veh.id_γ] = 0.0
                       xf[veh.id_γ] = 0.0
                       x0[veh.id_ψ] = 0.0
                       xf[veh.id_ψ] = 0.0
                       x = straightline_interpolate(x0, xf, N)

                       # Input guess
                       hover = zeros(pbm.nu)
                       hover[veh.id_T] = T_cst
                       hover[veh.id_δ] = 0.0
                       hover[veh.id_φ] = 0.0
                       u = straightline_interpolate(hover, hover, N)

                       return x, u, p
                       end)

    return nothing
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
    clr = get_colormap()(1.0)
    tf = sol.p[mdl.vehicle.id_t]
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

    # ..:: Acceleration bounds ::..
    bnd_max = mdl.vehicle.T_max*scale
    bnd_min = mdl.vehicle.T_min*scale
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # ..:: Thrust value (continuous-time) ::..
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    ct_thrust = T_RealVector([sample(sol.uc, τ)[mdl.vehicle.id_T]*scale
                              for τ in ct_τ])
    ax.plot(ct_time, ct_thrust,
            color=clr,
            linewidth=2)

    # ..:: Thrust value (discrete-time) ::..
    dt_time = sol.τd*sol.p[mdl.vehicle.id_t]
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
    tf = sol.p[mdl.vehicle.id_t]
    scale = 180/pi

    fig = create_figure((5, 5))

    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    dt_time = sol.τd*sol.p[mdl.vehicle.id_t]

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
        sample(sol.xc, τ)[mdl.vehicle.id_γ]*scale for τ in ct_τ])
    ax.plot(ct_time, ct_gimbal_delayed,
            color="#db6245",
            linestyle="--",
            linewidth=1,
            dash_capstyle="round")

    # >> Delayed gimbal angle (discrete-time) <<
    dt_gimbal_delayed = sol.xd[mdl.vehicle.id_γ, :]*scale
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
    ax.set_ylabel("Gimbal rate [\$^\\circ/s\$]")

    # >> Gimbal rate bounds <<
    pad = 2.0
    bnd_max = mdl.vehicle.β_max*scale
    bnd_min = -mdl.vehicle.β_max*scale
    y_top = bnd_max+pad
    y_bot = bnd_min-pad
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # >> Actual gimbal rate (discrete-time) <<
    δ = sol.ud[mdl.vehicle.id_δ, 1:end-1]
    δn = sol.ud[mdl.vehicle.id_δ, 2:end]
    dt_β = (δn-δ)./diff(dt_time)
    ax.plot(dt_time[2:end], dt_β*scale,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # >> Actual gimbal rate (continuous-time) <<
    β = T_ContinuousTimeTrajectory(sol.τd[1:end-1], dt_β, :zoh)
    ct_β = T_RealVector([sample(β, τ)*scale for τ in ct_τ])
    ax.plot(ct_time, ct_β,
            color=clr,
            linewidth=2,
            zorder=100)

    # >> Constraint (approximate) gimbal rate (discrete-time) <<
    δ = sol.xd[mdl.vehicle.id_γ, :]
    δn = sol.ud[mdl.vehicle.id_δ, :]
    dt_β = (δn-δ)./mdl.vehicle.rate_delay
    ax.plot(dt_time, dt_β*scale,
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

#= Plot the fin control (wind-facing area) trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution. =#
function plot_fin(mdl::StarshipProblem,
                  sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = sol.p[mdl.vehicle.id_t]
    scale = 1.0

    fig = create_figure((5, 5))

    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    dt_time = sol.τd*sol.p[mdl.vehicle.id_t]

    # ..:: Area timeseries ::..
    ax = fig.add_subplot(211)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Front fin area [m\$^2\$]")

    # >> Area bounds <<
    pad = 2.0
    bnd_max = 2*mdl.vehicle.Afin*scale
    bnd_min = 0.0
    y_top = bnd_max+pad
    y_bot = bnd_min-pad
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # >> Delayed area (continuous-time) <<
    ct_area_delayed = T_RealVector([
        2*sample(sol.xc, τ)[mdl.vehicle.id_ψ]*scale for τ in ct_τ])
    ax.plot(ct_time, ct_area_delayed,
            color="#db6245",
            linestyle="--",
            linewidth=1,
            dash_capstyle="round")

    # >> Delayed area (discrete-time) <<
    dt_area_delayed = 2*sol.xd[mdl.vehicle.id_ψ, :]*scale
    ax.plot(dt_time, dt_area_delayed,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0,
            markerfacecolor="#db6245",
            clip_on=false)

    # >> Area (continuous-time) <<
    ct_area = T_RealVector([
        2*sample(sol.uc, τ)[mdl.vehicle.id_φ]*scale for τ in ct_τ])
    ax.plot(ct_time, ct_area,
            color=clr,
            linewidth=2)

    # >> Area (discrete-time) <<
    dt_area = 2*sol.ud[mdl.vehicle.id_φ, :]*scale
    ax.plot(dt_time, dt_area,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # ..:: Area rate timeseries ::..
    ax = fig.add_subplot(212)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Area change rate [m\$^2\$/s]")

    # >> Area bounds <<
    pad = 2.0
    bnd_max = mdl.vehicle.α_max*scale
    bnd_min = -mdl.vehicle.α_max*scale
    y_top = bnd_max+pad
    y_bot = bnd_min-pad
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # >> Actual area (discrete-time) <<
    A = sol.ud[mdl.vehicle.id_φ, 1:end-1]
    An = sol.ud[mdl.vehicle.id_φ, 2:end]
    dt_α = (An-A)./diff(dt_time)
    ax.plot(dt_time[2:end], dt_α*scale,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # >> Actual area (continuous-time) <<
    α = T_ContinuousTimeTrajectory(sol.τd[1:end-1], dt_α, :zoh)
    ct_α = T_RealVector([sample(α, τ)*scale for τ in ct_τ])
    ax.plot(ct_time, ct_α,
            color=clr,
            linewidth=2,
            zorder=100)

    # >> Constraint (approximate) area (discrete-time) <<
    A = sol.xd[mdl.vehicle.id_ψ, :]
    An = sol.ud[mdl.vehicle.id_φ, :]
    dt_α = (An-A)./mdl.vehicle.rate_delay
    ax.plot(dt_time, dt_α*scale,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0,
            markerfacecolor="#db6245",
            clip_on=false,
            zorder=110)

    save_figure("starship_fin", algo)

    return nothing
end
