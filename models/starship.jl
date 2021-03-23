#= Starship landing flip maneuver data structures and custom methods.

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
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_θ::T_Int      # Tilt angle index of the state vector
    id_ω::T_Int      # Tilt rate index of the state vector
    id_δ::T_Int      # Gimbal angle index of the state vector
    id_T::T_Int      # Thrust index of the input vector
    id_α::T_Int      # Gimbal rate index of the input vector
    id_t::T_Int      # Index of time dilation
    T_max::T_Real    # [N] Maximum thrust
    T_min::T_Real    # [N] Minimum thrust
    δ_max::T_Real    # [rad] Maximum gimbal angle
    α_max::T_Real    # [rad/s] Maximum gimbal rate
    m::T_Real        # [kg] Vehicle mass
    J::T_Real        # [kg*m^2] Vehicle moment of inertia
    lg::T_Real       # [m] Distance from CG to engine gimbal
    ei::T_RealVector # Lateral body axis
    ej::T_RealVector # Longitudinal body axis
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
    id_δ = 7
    id_T = 1
    id_α = 2
    id_t = 1
    # >> Thrust bounds <<
    ne = 3 # Number of engines
    T_min1 = 880e3 # [N] One engine min thrust
    T_max1 = 2210e3 # [N] One engine max thrust
    T_max = ne*T_max1
    T_min = T_min1
    # >> Gimbal bounds <<
    δ_max = deg2rad(10.0)
    α_max = 2*δ_max
    # >> Mechanical properties <<
    H = 50.0 # [m] Stage height
    m = 120.0e3
    J = m/12*H^2
    lg = 0.5*H
    # >> Body frame <<
    ei = [1.0; 0.0]
    ej = [0.0; 1.0]

    starship = StarshipParameters(id_r, id_v, id_θ, id_ω, id_δ, id_T,
                                  id_α, id_t, T_max, T_min, δ_max, α_max,
                                  m, J, lg, ei, ej)

    # ..:: Environment ::..
    ex = [1.0; 0.0]
    ey = [0.0; 1.0]
    g = -9.81*ey
    env = StarshipEnvironmentParameters(ex, ey, g)

    # ..:: Trajectory ::..
    r0 = 100.0*ex+600.0*ey
    v0 = -75.0*ey
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
                       x0[veh.id_δ] = 0.0
                       xf[veh.id_δ] = 0.0
                       x = straightline_interpolate(x0, xf, N)

                       # Input guess
                       hover = zeros(pbm.nu)
                       hover[veh.id_T] = norm(veh.m*env.g)
                       hover[veh.id_α] = 0.0
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
    δ = sol.xd[mdl.vehicle.id_δ, :]
    θ = sol.xd[mdl.vehicle.id_θ, :]
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
        if altitude>100 || k==N || k in K
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
    y_top = mdl.vehicle.δ_max*scale+2.0
    y_bot = -y_top

    fig = create_figure((5, 5))

    # Plot data
    data = [Dict(:bnd=>mdl.vehicle.δ_max,
                 :pad=>2.0,
                 :ylabel=>"Gimbal angle [\$^\\circ\$]",
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>mdl.vehicle.id_δ),
            Dict(:bnd=>mdl.vehicle.α_max,
                 :pad=>2.0,
                 :ylabel=>"Gimbal rate [\$^\\circ/s\$]",
                 :dt_y=>sol.ud,
                 :ct_y=>sol.uc,
                 :id=>mdl.vehicle.id_α)]

    for i = 1:length(data)
        ax = fig.add_subplot(2, 1, i)

        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(data[i][:ylabel])

        # ..:: Acceleration bounds ::..
        bnd_max = data[i][:bnd]*scale
        bnd_min = -data[i][:bnd]*scale
        y_top = bnd_max+data[i][:pad]
        y_bot = bnd_min-data[i][:pad]
        plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
        plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

        # ..:: Thrust value (continuous-time) ::..
        ct_res = 500
        ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
        ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
        ct_gimbal = T_RealVector([
            sample(data[i][:ct_y], τ)[data[i][:id]]*scale for τ in ct_τ])
        ax.plot(ct_time, ct_gimbal,
                color=clr,
                linewidth=2)

        # ..:: Thrust value (discrete-time) ::..
        dt_time = sol.τd*sol.p[mdl.vehicle.id_t]
        dt_gimbal = data[i][:dt_y][data[i][:id], :]*scale
        ax.plot(dt_time, dt_gimbal,
                linestyle="none",
                marker="o",
                markersize=5,
                markeredgewidth=0,
                markerfacecolor=clr,
                clip_on=false,
                zorder=100)
    end

    save_figure("starship_gimbal", algo)

    return nothing
end
