"""
Starship landing plots.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

using PyPlot
using Colors

using Solvers

"""
    plot_trajectory_history(mdl, history)

Plot the trajectory evolution through SCP iterations.

Args:
- `mdl`: the quadrotor problem parameters.
- `history`: SCP iteration data history.
"""
function plot_trajectory_history(
        mdl::StarshipProblem,
        history::SCPHistory
)::Nothing

    # Common values
    num_iter = length(history.subproblems)
    algo = history.subproblems[1].algo
    _cmap = generate_colormap()
    cmap = v -> _cmap.to_rgba(v)[1:3]
    cmap_offset = 0.1
    alph_offset = 0.3

    fig = create_figure((3.4, 5))
    ax = fig.add_subplot()

    ax.axis("equal")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("Downrange [m]")
    ax.set_ylabel("Altitude [m]")

    # Draw the glide slope constraint
    _starship__plot_glideslope(ax, mdl)

    # ..:: Draw the trajectories ::..
    for i=0:num_iter
        # Extract values for the trajectory at iteration i
        if i==0
            trj = history.subproblems[1].ref
            alph = alph_offset
            clr = parse(RGB, "#356397")
            clr = rgb2pyplot(clr, a=alph)
            shp = "X"
        else
            trj = history.subproblems[i].sol
            f = (off) -> (i-1)/(num_iter-1)*(1-off)+off
            alph = f(alph_offset)
            clr = (cmap(f(cmap_offset))..., alph)
            shp = "o"
        end
        pos = trj.xd[mdl.vehicle.id_r, :]
        x, y = pos[1, :], pos[2, :]

        ax.plot(x, y,
                linestyle="none",
                marker=shp,
                markersize=5,
                markerfacecolor=clr,
                markeredgecolor=(1, 1, 1, alph),
                markeredgewidth=0.3,
                clip_on=false,
                zorder=100)
    end

    set_axis_equal(ax, (-200, missing, -25, 650))

    save_figure("starship_traj_iters.pdf", algo)

    return nothing
end

#= Plot the final converged trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution output by SCP. =#
function plot_final_trajectory(mdl::StarshipProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = generate_colormap().to_rgba(1.0)[1:3]
    N = size(sol.xd, 2)
    speed = [norm(sol.xd[mdl.vehicle.id_v, :][:, k]) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_cmap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)

    fig = create_figure((4, 5))
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

    # Draw the glide slope constraint
    _starship__plot_glideslope(ax, mdl)

    # ..:: Draw the final continuous-time position trajectory ::..
    # Collect the continuous-time trajectory data
    ct_res = 500
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    ct_pos = RealMatrix(undef, 2, ct_res)
    ct_speed = RealVector(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, ct_τ[k])
        ct_pos[:, k] = xk[mdl.vehicle.id_r[1:2]]
        ct_speed[k] = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res-1
        r, v = ct_pos[:, k], ct_speed[k]
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
    for k = 1:N
        altitude = dot(pos[:, k], mdl.env.ey)
        base = pos[:, k]
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

    set_axis_equal(ax, (-200, missing, -25, 650))

    save_figure("starship_final_traj.pdf", algo)

    return nothing
end

#= Plot the velocity trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution. =#
function plot_velocity(mdl::StarshipProblem,
                       sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    traj = mdl.traj
    clr = generate_colormap().to_rgba(1.0)[1:3]
    t1 = sol.p[veh.id_t1]
    t2 = sol.p[veh.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)
    xy_clrs = ["#db6245", "#5da9a1"]

    fig = create_figure((6, 3))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")

    # ..:: Velocity (continuous-time) ::..
    ct_res = 500
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    ct_vel = hcat([
        sample(sol.xc, τ)[mdl.vehicle.id_v] for τ in ct_τ]...)

    for i=1:2
        ax.plot(ct_time, ct_vel[i, :],
                color=xy_clrs[i],
                linewidth=2)
    end

    # ..:: Velocity (discrete-time) ::..
    dt_time = map(τ2t, sol.td)
    dt_vel = sol.xd[mdl.vehicle.id_v, :]
    for i=1:2
        ax.plot(dt_time, dt_vel[i, :],
                linestyle="none",
                marker="o",
                markersize=5,
                markeredgewidth=0,
                markerfacecolor=xy_clrs[i],
                clip_on=false,
                zorder=100)
    end

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

    save_figure("starship_velocity.pdf", algo)

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
    clr = generate_colormap().to_rgba(1.0)[1:3]
    t1 = sol.p[veh.id_t1]
    t2 = sol.p[veh.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)
    N = size(sol.xd, 2)
    scale = 1e-6
    y_top = 7.0
    y_bot = 0.0

    fig = create_figure((6, 3))
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
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    ct_thrust = RealVector([sample(sol.uc, τ)[mdl.vehicle.id_T]*scale
                              for τ in ct_τ])
    ax.plot(ct_time, ct_thrust,
            color=clr,
            linewidth=2)

    # ..:: Thrust value (discrete-time) ::..
    dt_time = map(τ2t, sol.td)
    dt_thrust = sol.ud[mdl.vehicle.id_T, :]*scale
    ax.plot(dt_time, dt_thrust,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

    save_figure("starship_thrust.pdf", algo)

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
    clr = generate_colormap().to_rgba(1.0)[1:3]
    t1 = sol.p[mdl.vehicle.id_t1]
    t2 = sol.p[mdl.vehicle.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    scale = 180/pi
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)

    fig = create_figure((6, 6))

    ct_res = 500
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    dt_time = map(τ2t, sol.td)

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
    ct_gimbal_delayed = RealVector([
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
    ct_gimbal = RealVector([
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

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

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
    δdot = ContinuousTimeTrajectory(sol.td[1:end-1], dt_δdot, :zoh)
    ct_δdot = RealVector([sample(δdot, τ)*scale for τ in ct_τ])
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

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

    save_figure("starship_gimbal.pdf", algo)

    return nothing
end

# ..:: Private methods ::..

""" Draw the glide slope constraint on an existing plot.

Args:
* `ax`: the figure axis object.
* `mdl`: the starship problem parameters.
* `alt`: (optional) altitude of glide slope "triangle" visualization.
"""
function _starship__plot_glideslope(ax::PyPlot.PyObject,
                                    mdl::StarshipProblem;
                                    alt::RealValue=200.0)::Nothing
    x_gs = alt*tan(mdl.traj.γ_gs)
    ax.plot([-x_gs, 0, x_gs], [alt, 0, alt],
            color="#5da9a1",
            linestyle="--",
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=90)
    return nothing
end

""" Draw the phase switch time on a timeseries plot.

Args:
* `ax`: the figure axis object.
* `t1`: the duration of phase 1.
"""
function _starship__plot_switch_time(ax::PyPlot.PyObject,
                                     t1::RealValue)::Nothing
    ax.axvline(x=t1,
               color="black",
               linestyle="--",
               linewidth=1,
               dash_capstyle="round")
    return nothing
end
