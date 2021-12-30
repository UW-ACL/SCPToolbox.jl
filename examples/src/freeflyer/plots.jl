"""
6-Degree of Freedom free-flyer problem plots.

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
    signed_distance(r, mdl)

Compute the signed distance function at the given position.

# Arguments
- `r`: the position vector.
- `mdl`: the free-flyer problem parameters.

# Returns
- `d`: the signed distance value.
"""
function signed_distance(
        r::RealVector,
        mdl::FreeFlyerProblem
)::Real
    room = mdl.env.iss
    d = logsumexp([1-norm((r-room[i].c)./room[i].s, Inf)
                   for i=1:mdl.env.n_iss]; t=mdl.traj.hom)
end


"""
    plot_zero_levelset(ax, mdl, z)

Plot the signed distance function zero-level set.

This gives a view of the space station flight space boundary that is seen by
the optimization, for a specific z-height.

# Arguments
    ax: the figure axis object.
    mdl: the free-flyer problem parameters.
    z: the z-coordinate at which to evaluate the signed distance function.
"""
function plot_zero_levelset(
        ax::PyPlot.PyObject,
        mdl::FreeFlyerProblem,
        z::Real
)::Nothing

    # Parameters
    xlims = (5.9, 12.1)
    ylims = (-2.6, 7.1)
    res = 100

    x = RealVector(LinRange(xlims..., res))
    y = RealVector(LinRange(ylims..., res))
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    f = (x, y) -> signed_distance([x; y; z], mdl)
    Z = map(f, X, Y)

    cs = ax.contour(x, y, Z, [0.0],
                    colors="#f1d46a",
                    linewidths=1,
                    linestyles="solid",
                    zorder=10)
    cs.collections[1].set_label("\$d_{\\mathrm{ss}}(r_{\\mathcal I})=0\$")

    return nothing
end

""" Plot the trajectory evolution through SCP iterations.

# Arguments
    mdl: the free-flyer problem parameters.
    history: SCP iteration data history.
"""
function plot_trajectory_history(mdl::FreeFlyerProblem,
                                 history::SCPHistory)::Nothing

    # Common values
    num_iter = length(history.subproblems)
    algo = history.subproblems[1].algo
    cmap = generate_colormap()
    cmap_offset = 0.1
    alph_offset = 0.3

    fig = create_figure((3, 4))
    ax = fig.add_subplot()

    ax.axis("equal")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.set_xlim((5.5, 12.5))

    ax.set_xlabel("Position \$r_{\\mathcal I,1}\$ [m]")
    ax.set_ylabel("Position \$r_{\\mathcal I,2}\$ [m]")

    plot_prisms!(ax, mdl.env.iss; label="\$\\mathcal O_i\$")
    plot_ellipsoids!(ax, mdl.env.obs; label="Obstacle")

    # ..:: Signed distance function zero-level set ::..
    z_iss = history.subproblems[end].sol.xd[mdl.vehicle.id_r[end], 1]
    plot_zero_levelset(ax, mdl, z_iss)

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
            label = "Initial \$r_{{\\mathcal I}}\$"
        elseif i == num_iter
            label = "Converged \$r_{{\\mathcal I}}\$"
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

    handles, labels = ax.get_legend_handles_labels()
    order = [1,2,5,3,4]
    leg = ax.legend(handles[order], labels[order],
                    framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (5.5, missing, -3, 7.5))

    save_figure("freeflyer_traj_iters.pdf", algo)

    return nothing
end


""" Plot the final converged trajectory.

# Arguments
    mdl: the free-flyer problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory(mdl::FreeFlyerProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = rgb(generate_colormap(), 1.0)
    N = size(sol.xd, 2)
    speed = [norm(sol.xd[mdl.vehicle.id_v, k]) for k=1:N]
    v_cmap = generate_colormap(
        "inferno";
        minval=minimum(speed),
        maxval=maximum(speed))
    u_scale = 8e1

    fig = create_figure((3.8, 4))
    ax = fig.add_subplot()

    ax.axis("equal")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.set_xlim((5.5, 12.5))

    ax.set_xlabel("Position \$r_{\\mathcal I,1}\$ [m]")
    ax.set_ylabel("Position \$r_{\\mathcal I,2}\$ [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity \$\\|v_{\\mathcal I}\\|_2\$ [m/s]")

    plot_prisms!(ax, mdl.env.iss; label="\$\\mathcal O_i\$")
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
                markersize=3,
                markerfacecolor=v_cmap.to_rgba(v),
                markeredgecolor="none",
                alpha=0.2,
                zorder=100)
    end

    # ..:: Draw the discrete-time positions trajectory ::..
    pos = sol.xd[mdl.vehicle.id_r, :]
    x, y = pos[1, :], pos[2, :]
    ax.plot(x, y,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label="\$r_{\\mathcal I}\$",
            zorder=100)

    # ..:: Draw the thrust vector ::..
    thrust = sol.ud[mdl.vehicle.id_T, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    for k = 1:N
        base = pos[1:2, k]
        tip = base+u_scale*thrust[1:2, k]
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#db6245",
                linewidth=1.2,
                solid_capstyle="round",
                label=((k==1) ? "\$T_{\\mathcal I}\$ (scaled)" :
                       nothing),
                zorder=99)
    end

    # ..:: Signed distance function zero-level set ::..
    z_iss = sol.xd[mdl.vehicle.id_r[end], 1]
    plot_zero_levelset(ax, mdl, z_iss)

    handles, labels = ax.get_legend_handles_labels()
    order = [1,2,5,3,4]
    leg = ax.legend(handles[order], labels[order],
                    framealpha=0.8, fontsize=8, loc="upper left")
    leg.set_zorder(200)

    set_axis_equal(ax, (5.5, missing, -3, 7.5))

    save_figure("freeflyer_final_traj.pdf", algo)

    return nothing
end


""" Timeseries signal plots.

# Arguments
    mdl: the free-flyer problem parameters.
    sol: the trajectory solution.
"""
function plot_timeseries(mdl::FreeFlyerProblem,
                         sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    ct_res = 500
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    tf = sol.p[veh.id_t]
    dt_time = sol.td*tf
    ct_time = ct_τ*tf
    clr = rgb(generate_colormap(), 1.0)
    xyz_clrs = ["#db6245", "#5da9a1", "#356397"]
    marker_darken_factor = 0.2
    top_scale = 1.1

    fig = create_figure((6, 6))

    # Plot data
    data = [Dict(:y_top=>top_scale*veh.T_max*1e3,
                 :bnd_max=>veh.T_max,
                 :ylabel=>"Thrust \$T_{\\mathcal{I}}\$ [mN]",
                 :legend=>vcat([@sprintf("\$T_{\\mathcal{I},%d}\$",i)
                                for i=1:3],
                               "\$\\|T_{\\mathcal{I}}\\|_2\$"),
                 :scale=>(T)->T*1e3,
                 :dt_y=>sol.ud,
                 :ct_y=>sol.uc,
                 :id=>veh.id_T),
            Dict(:y_top=>top_scale*veh.M_max*1e6,
                 :bnd_max=>veh.M_max,
                 :ylabel=>"Torque \$M_{\\mathcal{I}}\$ [\$\\mu\$N\$\\cdot\$m]",
                 :legend=>vcat([@sprintf("\$M_{\\mathcal{I},%d}\$",i)
                                for i=1:3],
                               "\$\\|M_{\\mathcal{I}}\\|_2\$"),
                 :scale=>(M)->M*1e6,
                 :dt_y=>sol.ud,
                 :ct_y=>sol.uc,
                 :id=>veh.id_M),
            Dict(:y_top=>nothing,
                 :bnd_max=>nothing,
                 :ylabel=>string("Attitude \$q_{\\scriptscriptstyle",
                                 "\\mathcal{B}\\gets\\mathcal{I}}\$ as ZYX\n",
                                 "Euler angles [\$^\\circ\$]"),
                 :legend=>["\$\\varphi\$", "\$\\theta\$", "\$\\psi\$"],
                 :scale=>(q)->rad2deg.(collect(rpy(Quaternion(q)))),
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_q),
            Dict(:y_top=>top_scale*rad2deg(veh.ω_max),
                 :bnd_max=>veh.ω_max,
                 :ylabel=>"Angular rate \$\\omega_{\\mathcal{B}}\$ [\$^\\circ\$/s]",
                 :legend=>vcat([@sprintf("\$\\omega_{\\mathcal{B},%d}\$",i) for i=1:3],
                               "\$\\|\\omega_{\\mathcal{B}}\\|_2\$"),
                 :scale=>(ω)->rad2deg.(ω),
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_ω)]

    axes = []

    for i = 1:length(data)
        ax = fig.add_subplot(2, 2, i)
        push!(axes, ax)

        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(data[i][:ylabel])

        y_top = data[i][:y_top]
        y_max = data[i][:bnd_max]
        if !isnothing(y_max)
            y_max = data[i][:scale](y_max)
        end

        if !isnothing(y_max)
            plot_timeseries_bound!(ax, 0.0, tf, y_max, y_top-y_max)
        end

        # >> Continuous-time components <<
        yc = hcat([data[i][:scale](sample(data[i][:ct_y], τ)[data[i][:id]])
                   for τ in ct_τ]...)

        for j = 1:3
            ax.plot(ct_time, yc[j, :],
                    color=xyz_clrs[j],
                    linewidth=1,
                    zorder=100)
        end

        # >> Discrete-time components <<
        yd = data[i][:dt_y][data[i][:id], :]
        yd = hcat([data[i][:scale](yd[:, k]) for k=1:size(yd,2)]...)

        for j = 3:-1:1
            local mark_clr = weighted_color_mean(1-marker_darken_factor,
                                                 parse(RGB, xyz_clrs[j]),
                                                 colorant"black")

            for visible in [true, false]
                ax.plot(visible ? dt_time : [],
                        visible ? yd[j, :] : [],
                        linestyle=visible ? "none" : "-",
                        color=visible ? nothing : xyz_clrs[j],
                        linewidth=1,
                        solid_capstyle="round",
                        marker="o",
                        markersize=3,
                        markeredgewidth=0.0,
                        markerfacecolor=rgb2pyplot(mark_clr),
                        clip_on=!visible,
                        zorder=100-Int(!visible)*10,
                        label=visible ? nothing : data[i][:legend][j])
            end
        end

        # >> Continuous-time norm <<
        if !isnothing(y_max)
            y_nrm = RealVector([norm(yc[:, k]) for k in 1:ct_res])
            ax.plot(ct_time, y_nrm,
                    color=clr,
                    linewidth=2,
                    linestyle=":",
                    dash_capstyle="round",
                    zorder=100,
                    label=data[i][:legend][4])
        end

        tf_max = round(tf, digits=5)
        ax.set_xlim((0.0, tf_max))
        ax.set_xticks(round.(Int, LinRange(0, tf_max, 5)))
        ax.set_ylim((minimum(yc), isnothing(y_top) ? maximum(yc) : y_top))

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(reverse(handles), reverse(labels),
                        framealpha=0.8, fontsize=8, loc="upper right")
        leg.set_zorder(200)
    end

    fig.align_ylabels(axes[[1,3]])
    fig.align_ylabels(axes[[2,4]])

    save_figure("freeflyer_timeseries.pdf", algo)

    return nothing
end

""" Timeseries plot of obstacle constraint values for final trajectory.

# Arguments
- `mdl`: the free-flyer problem parameters.
- `sol`: the trajectory solution.
"""
function plot_obstacle_constraints(mdl::FreeFlyerProblem,
                                   sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    env = mdl.env
    ct_res = 500
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    tf = sol.p[veh.id_t]
    dt_time = sol.td*tf
    ct_time = ct_τ*tf
    cmap = generate_colormap()
    xyz_clrs = ["#db6245", "#5da9a1", "#356397"]
    marker_darken_factor = 0.2

    fig = create_figure((5.8, 2.8))

    # ..:: Plot ISS flight space constraint ::..
    ax = fig.add_subplot(1, 2, 1)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("SDF \$\\tilde d_{\\mathrm{ss}}(r_{\\mathcal{I}})\$")

    # >> Continuous-time components <<
    yc = RealVector([signed_distance(
        sample(sol.xc, τ)[veh.id_r], mdl) for τ in ct_τ])
    y_bot = min(-0.1, minimum(yc))
    plot_timeseries_bound!(ax, 0.0, tf, 0.0, y_bot)

    ax.plot(ct_time, yc,
            color=rgb(cmap, 1.0),
            linewidth=1)

    # >> Discrete-time components <<
    yd = sol.xd[veh.id_r, :]
    yd = RealVector([signed_distance(yd[:, k], mdl)
                       for k=1:size(yd, 2)])
    ax.plot(dt_time, yd,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0.0,
            markerfacecolor=rgb(cmap, 1.0),
            clip_on=false,
            zorder=100)

    tf_max = round(tf, digits=5)
    ax.set_xlim((0.0, tf_max))
    ax.set_xticks(round.(Int, LinRange(0, tf_max, 5)))
    ax.set_ylim((y_bot, maximum(yc)))

    # ..:: Plot ellipsoid obstacle constraints ::..
    ax = fig.add_subplot(1, 2, 2)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("\$\\| H_j(r_{\\mathcal{I}}-c_j)\\|_2\$")

    clr_offset = 0.4
    cval = (j) -> (env.n_obs==1) ? 1.0 :
        (j-1)/(env.n_obs-1)*(1-clr_offset)+clr_offset

    plot_timeseries_bound!(ax, 0.0, tf, 1.0, -1.0)

    # >> Continuous-time components <<
    for j = 1:env.n_obs
        yc = RealVector([env.obs[j](sample(sol.xc, τ)[veh.id_r])
                           for τ in ct_τ])

        ax.plot(ct_time, yc,
                color=rgb(cmap, cval(j)),
                linewidth=1)
    end

    # >> Discrete-time components <<
    y_top = -Inf
    for j = env.n_obs:-1:1
        yd = sol.xd[veh.id_r, :]
        yd = RealVector([env.obs[j](yd[:, k]) for k=1:size(yd, 2)])
        y_top = max(y_top, maximum(yd))

        local mark_clr = weighted_color_mean(1-marker_darken_factor,
                                             RGB(rgb(cmap, cval(j))...),
                                             colorant"black")

        for visible in [true, false]
            ax.plot(visible ? dt_time : [],
                    visible ? yd : [],
                    linestyle=visible ? "none" : "-",
                    color=visible ? nothing : rgb(cmap, cval(j)),
                    linewidth=1,
                    solid_capstyle="round",
                    marker="o",
                    markersize=3,
                    markeredgewidth=0.0,
                    markerfacecolor=rgb2pyplot(mark_clr),
                    clip_on=!visible,
                    zorder=100-Int(!visible)*101,
                    label=(visible ? nothing :
                           @sprintf("Obstacle \$j=%d\$", j)))
        end
    end

    tf_max = round(tf, digits=5)
    ax.set_xlim((0.0, tf_max))
    ax.set_xticks(round.(Int, LinRange(0, tf_max, 5)))
    ax.set_ylim((0.0, y_top))

    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(reverse(handles), reverse(labels),
                    framealpha=0.8, fontsize=8, loc="upper center")
    leg.set_zorder(200)

    save_figure("freeflyer_obstacles.pdf", algo)

    return nothing
end
