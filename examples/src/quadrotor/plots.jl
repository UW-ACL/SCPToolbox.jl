"""
Quadrotor obstacle avoidance plots.

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
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

using PyPlot
using Colors

using Solvers

"""
    plot_trajectory_history(mdl, history)

Plot the trajectory evolution through SCP iterations.

# Arguments
- `mdl`: the quadrotor problem parameters.
- `history`: SCP iteration data history.
"""
function plot_trajectory_history(
        mdl::QuadrotorProblem,
        history::SCPHistory
)::Nothing

    # Common values
    num_iter = length(history.subproblems)
    algo = history.subproblems[1].algo
    cmap = generate_colormap()
    cmap_offset = 0.1
    alph_offset = 0.3

    fig = create_figure((2.58, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position \$r_1\$ [m]")
    ax.set_ylabel("North position \$r_2\$ [m]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    # ..:: Draw the trajectories ::..
    for i = 0:num_iter
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
            clr = (rgb(cmap, f(cmap_offset))..., alph)
            shp = "o"
        end
        pos = trj.xd[mdl.vehicle.id_r, :]
        x, y = pos[1, :], pos[2, :]

        label = nothing
        if i == 0
            label = "Initial \$r\$"
        elseif i == num_iter
            label = "Converged \$r\$"
        end

        ax.plot(x, y,
                linestyle="none",
                marker=shp,
                markersize=5,
                markerfacecolor=clr,
                markeredgecolor=(1, 1, 1, alph),
                markeredgewidth=0.3,
                label=label,
                zorder=100)
    end

    ax.set_xticks(-0.5:1:5)

    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (-0.5, missing, -0.5, 6.5))

    save_figure("quadrotor_traj_iters.pdf", algo)

    return nothing
end

"""
    plot_final_trajectory(mdl, sol)

Plot the final converged trajectory.

# Arguments
- `mdl`: the quadrotor problem parameters.
- `sol`: the trajectory solution.
"""
function plot_final_trajectory(
        mdl::QuadrotorProblem,
        sol::SCPSolution
)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = rgb(generate_colormap(), 1.0)
    N = size(sol.xd, 2)
    speed = [norm(sol.xd[mdl.vehicle.id_v, k]) for k=1:N]
    v_cmap = generate_colormap(
        "inferno";
        minval=minimum(speed),
        maxval=maximum(speed)
    )
    u_scale = 0.2

    fig = create_figure((3.27, 4))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position \$r_1\$ [m]")
    ax.set_ylabel("North position \$r_2\$ [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity \$\\|\\dot r\\|_2\$ [m/s]")

    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

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
            label="\$r\$",
            zorder=100)

    # ..:: Draw the acceleration vector ::..
    acc = sol.ud[mdl.vehicle.id_u, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    for k = 1:N
        base = pos[1:2, k]
        tip = base+u_scale*acc[1:2, k]
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                label=(k==1) ? "\$a\$ (scaled)" : nothing,
                zorder=99)
    end

    ax.set_xticks(-0.5:1:5)

    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (-0.5, missing, -0.5, 6.5))

    save_figure("quadrotor_final_traj.pdf", algo)

    return nothing
end

"""
    plot_input_norm(mdl, sol)

Plot the acceleration input norm.

# Arguments
- `mdl`: the quadrotor problem parameters.
- `sol`: the trajectory solution.
"""
function plot_input_norm(
        mdl::QuadrotorProblem,
        sol::SCPSolution
)::Nothing

    # Common
    algo = sol.algo
    clr = rgb(generate_colormap(), 1.0)
    tf = sol.p[mdl.vehicle.id_t]
    y_top = 25.0
    y_bot = 0.0

    fig = create_figure((5, 2.75))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration \$\\|a\\|_2\$ [m/s\$^2\$]")

    # ..:: Acceleration bounds ::..
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # ..:: Norm of acceleration vector (continuous-time) ::..
    ct_res = 500
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    ct_acc_vec = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_acc_nrm = RealVector([norm(ct_acc_vec[:, k]) for k in 1:ct_res])
    ax.plot(ct_time, ct_acc_nrm,
            color=clr,
            linewidth=2)

    # ..:: Norm of acceleration vector (discrete-time) ::..
    time = sol.td*sol.p[mdl.vehicle.id_t]
    acc_vec = sol.ud[mdl.vehicle.id_u, :]
    acc_nrm = RealVector([norm(acc_vec[:, k]) for k in 1:size(acc_vec, 2)])
    for visible in [true, false]
        ax.plot(visible ? time : [],
                visible ? acc_nrm : [],
                linestyle=visible ? "none" : "-",
                color=visible ? nothing : clr,
                linewidth=2,
                marker="o",
                markersize=5,
                markeredgewidth=0,
                markerfacecolor=clr,
                zorder=100-Int(!visible)*200,
                clip_on=!visible,
                label=visible ? nothing : "\$\\|a\\|_2\$")
    end

    # ..:: Slack input (discrete-time) ::..
    σ = sol.ud[mdl.vehicle.id_σ, :]
    ax.plot(time, σ,
            linestyle="none",
            marker="h",
            markersize=4,
            markeredgecolor=clr,
            markeredgewidth=0.3,
            markerfacecolor="#f1d46a",
            clip_on=false,
            zorder=100,
            label="\$\\sigma\$")

    leg = ax.legend(framealpha=0.8, fontsize=8, loc="upper right")
    leg.set_zorder(200)

    tf_max = round(tf, digits=5)
    ax.set_xlim((0.0, tf_max))
    ax.set_xticks(LinRange(0, tf_max, 6))

    save_figure("quadrotor_input.pdf", algo)

    return nothing
end

"""
    plot_tilt_angle(mdl, sol)

Plot the acceleration input norm.

# Arguments
    mdl: the quadrotor problem parameters.
    sol: the trajectory solution.
"""
function plot_tilt_angle(
        mdl::QuadrotorProblem,
        sol::SCPSolution
)::Nothing

    # Common
    algo = sol.algo
    clr = rgb(generate_colormap(), 1.0)
    tf = sol.p[mdl.vehicle.id_t]
    y_top = 70.0

    fig = create_figure((5, 2.75))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)
    ax.set_ylim((0, y_top))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(string("Tilt",
                         " \$\\arccos({\\hat n^{\\scriptscriptstyle",
                         "\\mathsf{T}}a}\\|a\\|_2^{-1})\$",
                         " [\$^\\circ\$]"))

    # ..:: Tilt angle bounds ::..
    bnd_max = rad2deg(mdl.vehicle.tilt_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)

    # ..:: Tilt angle (continuous-time) ::..
    ct_res = 500
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    _u = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_tilt = RealVector([acosd(_u[3, k]/norm(_u[:, k])) for k in 1:ct_res])
    ax.plot(ct_time, ct_tilt,
            color=clr,
            linewidth=2)

    # ..:: Tilt angle (discrete-time) ::..
    time = sol.td*sol.p[mdl.vehicle.id_t]
    _u = sol.ud[mdl.vehicle.id_u, :]
    tilt = RealVector([acosd(_u[3, k]/norm(_u[:, k])) for k in 1:size(_u, 2)])
    ax.plot(time, tilt,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    tf_max = round(tf, digits=5)
    ax.set_xlim((0.0, tf_max))
    ax.set_xticks(LinRange(0, tf_max, 6))

    save_figure("quadrotor_tilt.pdf", algo)

    return nothing
end
