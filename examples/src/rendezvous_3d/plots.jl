#= Spacecraft rendezvous plots.

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
this program.  If not, see <https://www.gnu.org/licenses/>. =#

LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("parameters.jl")

    # include("../../../solvers/src/Solvers.jl")

    # using .Solvers
end

using PyPlot
using Printf

using Solvers

# ..:: Globals ::..

const LineCollection = PyPlot.matplotlib.collections.LineCollection

# ..:: Methods ::..

"""
    plot_trajectory_2d(mdl, sol)

Plot the final converged trajectory, projected onto 2D planes.

# Arguments
- `mdl`: the problem description object.
- `sol`: the problem solution.
"""
function plot_trajectory_2d(mdl::RendezvousProblem,
                            sol::SCPSolution)::Nothing #noerr

    # Parameters
    algo = sol.algo
    veh = mdl.vehicle
    axis_inf_scale = 1e4
    pad_x = 0.1
    pad_y = 0.1
    equal_ar = true # Equal aspect ratio
    input_scale = 0.2

    # Plotting data
    ct_res = 1000
    dt_τ = sol.td
    dt_res = length(sol.td)
    ct_τ = RealVector(LinRange(0, 1, ct_res))
    tf = sol.p[veh.id_t]
    ct_t = ct_τ*tf
    dt_pos = sol.xd[veh.id_r, :]
    ct_pos = hcat([sample(sol.xc, τ)[veh.id_r] for τ in ct_τ]...) #noerr
    ct_vel = hcat([sample(sol.xc, τ)[veh.id_v] for τ in ct_τ]...) #noerr
    ct_speed = squeeze(mapslices(norm, ct_vel, dims=(1)))
    dt_thrust = sol.ud[veh.id_T, :]

    # Colormap
    v_cmap = generate_colormap("inferno";
                               minval=minimum(ct_speed),
                               maxval=maximum(ct_speed))

    data = [Dict(:prj=>[1, 3],
                 :x_name=>"x",
                 :y_name=>"z"),
            Dict(:prj=>[1, 2],
                 :x_name=>"x",
                 :y_name=>"y"),
            Dict(:prj=>[2, 3],
                 :x_name=>"y",
                 :y_name=>"z")]

    fig = create_figure((6, 10))

    num_plts = length(data)
    gspec = fig.add_gridspec(ncols=1, nrows=num_plts+1,
                             height_ratios=[0.05, fill(1, num_plts)...])

    axes = []

    for i_plt = 1:num_plts

        # Data
        prj = data[i_plt][:prj]
        ax_x_name = data[i_plt][:x_name]
        ax_y_name = data[i_plt][:y_name]

        ax = setup_axis!(gspec[i_plt+1, 1];
                         xlabel=@sprintf("LVLH \$%s\$ position [m]",
                                         ax_x_name),
                         ylabel=@sprintf("LVLH \$%s\$ position [m]",
                                         ax_y_name))
        push!(axes, ax)

        # Trajectory projection and bounding box
        ct_pos_2d = ct_pos[prj, :]
        dt_pos_2d = dt_pos[prj, :]
        dt_thrust_2d = dt_thrust[prj, :]

        bbox = Dict(:x=>Dict(:min=>minimum(ct_pos_2d[1, :]),
                             :max=>maximum(ct_pos_2d[1, :])),
                    :y=>Dict(:min=>minimum(ct_pos_2d[2, :]),
                             :max=>maximum(ct_pos_2d[2, :])))
        padding = Dict(:x=>pad_x*(bbox[:x][:max]-bbox[:x][:min]),
                       :y=>pad_x*(bbox[:y][:max]-bbox[:y][:min]))

        # Plot axes "guidelines"
        origin = [0; 0]
        xtip = [1; 0]
        ytip = [0; 1]
        ax.plot([origin[1], xtip[1]*axis_inf_scale],
                [origin[2], xtip[2]*axis_inf_scale],
                color=DarkBlue, #noerr
                linewidth=0.5,
                solid_capstyle="round")
        ax.plot([origin[1], ytip[1]*axis_inf_scale],
                [origin[2], ytip[2]*axis_inf_scale],
                color=DarkBlue, #noerr
                linewidth=0.5,
                solid_capstyle="round")

        # Plot the conitnuous-time trajectory
        line_segs = Vector{RealMatrix}(undef, 0)
        line_clrs = Vector{NTuple{4, RealValue}}(undef, 0)
        overlap = 3
        for k=1:ct_res-overlap
            push!(line_segs, ct_pos_2d[:, k:k+overlap]')
            push!(line_clrs, v_cmap.to_rgba(ct_speed[k]))
        end
        traj = LineCollection(line_segs, zorder=10,
                              colors = line_clrs,
                              linewidths=3)
        ax.add_collection(traj)

        # Plot the discrete-time trajectory
        ax.scatter(dt_pos_2d[1, :], dt_pos_2d[2, :],
                   marker="o",
                   c=DarkBlue, #noerr
                   s=5,
                   edgecolors="white",
                   linewidths=0.2,
                   zorder=20)

        # Axis limits
        xmin = bbox[:x][:min]-padding[:x]
        xmax = bbox[:x][:max]+padding[:x]
        ymin = bbox[:y][:min]-padding[:y]
        ymax = bbox[:y][:max]+padding[:y]

        # Detect zero-range axes
        if (xmax-xmin)<=1e-5
            xmin, xmax = -1, 1
        end
        if (ymax-ymin)<=1e-5
            ymin, ymax = -1, 1
        end

        if !equal_ar
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
        else
            # First, try to leave ymax unconstrained
            # Check to make sure that all y values contained in the resulting
            # box
            set_axis_equal(ax, (xmin, xmax, ymin, missing))
            y_rng = collect(ax.get_ylim())
            y_mid = sum(y_rng)/2
            y_rng .-= y_mid
            ymin = y_rng[1]
            set_axis_equal(ax, (xmin, xmax, ymin, missing))
            y_rng = collect(ax.get_ylim())
            if (any(dt_pos_2d[2, :].>y_rng[2]) ||
                any(dt_pos_2d[2, :].<y_rng[1]))
                # The data does not fit, leave xmax unconstrained instead
                set_axis_equal(ax, (xmin, missing, ymin, ymax))
                x_rng = collect(ax.get_xlim())
                x_mid = sum(x_rng)/2
                x_rng .-= x_mid
                xmin = x_rng[1]
                set_axis_equal(ax, (xmin, missing, ymin, ymax))
            end
        end
        x_rng = collect(ax.get_xlim())
        y_rng = collect(ax.get_ylim())

        # Thrust directions
        ref_sz = min(x_rng[2]-x_rng[1], y_rng[2]-y_rng[1])
        max_sz = maximum(mapslices(norm, dt_thrust_2d, dims=1))
        scale_factor = ref_sz/max_sz*input_scale
        thrust_segs = Vector{RealMatrix}(undef, 0)
        for k=1:dt_res
            thrust_whisker_base = dt_pos_2d[:, k]
            thrust_whisker_tip = thrust_whisker_base+
                scale_factor*dt_thrust_2d[:, k]
            push!(thrust_segs, hcat(thrust_whisker_base,
                                    thrust_whisker_tip)')
        end
        thrusts = LineCollection(thrust_segs, zorder=15,
                                 colors=Red, #noerr
                                 linewidths=1.5)
        thrusts.set_capstyle("round")
        ax.add_collection(thrusts)

        # Label the LVLH axes
        padding_x = pad_x*(x_rng[2])
        padding_y = pad_y*(y_rng[2])
        xlabel_pos = xtip*(x_rng[2]-padding_x)
        ylabel_pos = ytip*(y_rng[2]-padding_y)
        ax.text(xlabel_pos[1], xlabel_pos[2],
                @sprintf("\$\\hat %s\$", ax_x_name),
                ha="center",
                va="center",
                backgroundcolor="white")
        ax.text(ylabel_pos[1], ylabel_pos[2],
                @sprintf("\$\\hat %s\$", ax_y_name),
                ha="center",
                va="center",
                backgroundcolor="white")

        ax.invert_yaxis()

    end

    fig.align_ylabels(axes)

    # Colorbar
    cbar_ax = fig.add_subplot(gspec[1, 1])
    fig.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity \$\\|v\\|_2\$ [m/s]",
                 orientation="horizontal",
                 cax=cbar_ax)
    cbar_ax.xaxis.set_label_position("top")
    cbar_ax.xaxis.set_ticks_position("top")
    ax_pos = cbar_ax.get_position()
    ax_pos = [ax_pos.x0, ax_pos.y0-0.05, ax_pos.width, ax_pos.height]
    cbar_ax.set_position(ax_pos)

    save_figure("rendezvous_trajectory_2d", algo)

    return nothing
end # function

"""
    plot_state_timeseries(mdl, sol)

Plot the state component evolution as a function of time.

# Arguments
- `mdl`: the problem description object.
- `sol`: the problem solution.
"""
function plot_state_timeseries(mdl::RendezvousProblem,
                               sol::SCPSolution)::Nothing #noerr

    # Parameters
    algo = sol.algo
    veh = mdl.vehicle
    marker_darken_factor = 0.3
    outline_w = 1.5

    y1_clr = Blue #noerr
    y2_clr = Red #noerr
    darker_y1_clr = darken_color(y1_clr, marker_darken_factor)
    darker_y2_clr = darken_color(y2_clr, marker_darken_factor)

    # >> Plotting data <<
    ct_res = 1000
    dt_τ = sol.td
    ct_τ = RealVector(LinRange(0, 1, ct_res))
    tf = sol.p[veh.id_t]
    ct_t = ct_τ*tf
    dt_t = dt_τ*tf

    ct_r = hcat([sample(sol.xc, τ)[veh.id_r] for τ in ct_τ]...) #noerr
    ct_v = hcat([sample(sol.xc, τ)[veh.id_v] for τ in ct_τ]...) #noerr
    ct_q = hcat([sample(sol.xc, τ)[veh.id_q] for τ in ct_τ]...) #noerr
    ct_rpy = mapslices(q->collect(rpy(Quaternion(q))), ct_q, dims=1) #noerr
    ct_ω = hcat([sample(sol.xc, τ)[veh.id_ω] for τ in ct_τ]...) #noerr

    dt_r = sol.xd[veh.id_r, :]
    dt_v = sol.xd[veh.id_v, :]
    dt_q = sol.xd[veh.id_q, :]
    dt_rpy = mapslices(q->collect(rpy(Quaternion(q))), dt_q, dims=1) #noerr
    dt_ω = sol.xd[veh.id_ω, :]

    ct_rpy = rad2deg.(ct_rpy)
    dt_rpy = rad2deg.(dt_rpy)
    ct_ω = rad2deg.(ct_ω)
    dt_ω = rad2deg.(dt_ω)

    data=[Dict(:y1_dt=>dt_r[1, :],
               :y1_ct=>ct_r[1, :],
               :y2_dt=>dt_v[1, :],
               :y2_ct=>ct_v[1, :],
               :y1_label=>"LVLH \$x\$ position [m]",
               :y2_label=>"LVLH \$x\$ velocity [m/s]"),
          Dict(:y1_dt=>dt_r[2, :],
               :y1_ct=>ct_r[2, :],
               :y2_dt=>dt_v[2, :],
               :y2_ct=>ct_v[2, :],
               :y1_label=>"LVLH \$y\$ position [m]",
               :y2_label=>"LVLH \$y\$ velocity [m/s]"),
          Dict(:y1_dt=>dt_r[3, :],
               :y1_ct=>ct_r[3, :],
               :y2_dt=>dt_v[3, :],
               :y2_ct=>ct_v[3, :],
               :y1_label=>"LVLH \$z\$ position [m]",
               :y2_label=>"LVLH \$z\$ velocity [m/s]",
               :x_label=>"Time [s]"),
          Dict(:y1_dt=>dt_rpy[3, :],
               :y1_ct=>ct_rpy[3, :],
               :y2_dt=>dt_ω[1, :],
               :y2_ct=>ct_ω[1, :],
               :y1_label=>"Roll angle [\$^\\circ\$]",
               :y2_label=>"Body \$x\$ angular rate [\$^\\circ\$/s]"),
          Dict(:y1_dt=>dt_rpy[2, :],
               :y1_ct=>ct_rpy[2, :],
               :y2_dt=>dt_ω[2, :],
               :y2_ct=>ct_ω[2, :],
               :y1_label=>"Pitch angle [\$^\\circ\$]",
               :y2_label=>"Body \$y\$ angular rate [\$^\\circ\$/s]"),
          Dict(:y1_dt=>dt_rpy[1, :],
               :y1_ct=>ct_rpy[1, :],
               :y2_dt=>dt_ω[3, :],
               :y2_ct=>ct_ω[3, :],
               :y1_label=>"Yaw angle [\$^\\circ\$]",
               :y2_label=>"Body \$z\$ angular rate [\$^\\circ\$/s]",
               :x_label=>"Time [s]")]

    # >> Initialize figure <<
    fig = create_figure((10, 10))
    gspec = fig.add_gridspec(ncols=2, nrows=3)
    id_splot = [1 2;3 4;5 6]

    ax_list = []
    ax2_list = []

    for i = 1:length(data)

        # Data
        y1_data_dt = data[i][:y1_dt]
        y1_data_ct = data[i][:y1_ct]
        y2_data_dt = data[i][:y2_dt]
        y2_data_ct = data[i][:y2_ct]

        j = id_splot[i]
        ax = setup_axis!(gspec[j],
                         tight="both")
        ax2 = ax.twinx()

        push!(ax_list, ax)
        push!(ax2_list, ax2)

        if :x_label in keys(data[i])
            ax.set_xlabel(data[i][:x_label])
        else
            ax.tick_params(axis="x", which="both", bottom=false, top=false,
                           labelbottom=false)
        end

        ax.set_ylabel(data[i][:y1_label], color=y1_clr)
        ax.tick_params(axis="y", colors=y1_clr)
        ax.spines["left"].set_edgecolor(y1_clr)

        ax2.set_ylabel(data[i][:y2_label], color=y2_clr)
        ax2.tick_params(axis="y", colors=y2_clr)
        ax2.spines["right"].set_edgecolor(y2_clr)
        ax2.spines["top"].set_visible(false)
        ax2.spines["bottom"].set_visible(false)
        ax2.spines["left"].set_visible(false)
        ax2.set_axisbelow(true)

        # Continuous-time trajectory
        ax.plot(ct_t, y1_data_ct,
                color=y1_clr, #noerr
                linewidth=2,
                solid_joinstyle="round",
                solid_capstyle="round",
                clip_on=false,
                zorder=10)

        ax2.plot(ct_t, y2_data_ct,
                 color=y2_clr, #noerr
                 linewidth=2,
                 solid_joinstyle="round",
                 solid_capstyle="round",
                 clip_on=false,
                 zorder=10)
        ax2.plot(ct_t, y2_data_ct,
                 color="white",
                 linewidth=2+outline_w,
                 solid_joinstyle="round",
                 solid_capstyle="round",
                 clip_on=false,
                 zorder=9)

        # Discrete-time trajectory
        ax.plot(dt_t, y1_data_dt,
                linestyle="none",
                marker="o",
                markersize=3.5,
                markerfacecolor=darker_y1_clr, #noerr
                markeredgecolor="white",
                markeredgewidth=0.2,
                clip_on=false,
                zorder=20)

        ax2.plot(dt_t, y2_data_dt,
                 linestyle="none",
                 marker="o",
                 markersize=3.5,
                 markerfacecolor=darker_y2_clr, #noerr
                 markeredgecolor="white",
                 markeredgewidth=0.2,
                 clip_on=false,
                 zorder=20)
        ax2.plot(dt_t, y2_data_dt,
                 linestyle="none",
                 marker="o",
                 markersize=3.5+outline_w,
                 markerfacecolor="white",
                 markeredgewidth=0,
                 clip_on=false,
                 zorder=9)

    end

    fig.align_ylabels(ax_list[1:3])
    fig.align_ylabels(ax2_list[1:3])
    fig.align_ylabels(ax_list[4:6])
    fig.align_ylabels(ax2_list[4:6])

    save_figure("rendezvous_timeseries", algo)

    return nothing
end # function

"""
    plot_control(mdl, sol)

Plot the control inputs versus time.

# Arguments
- `mdl`: the problem description object.
- `sol`: the problem solution.
"""
function plot_inputs(mdl::RendezvousProblem,
                     sol::SCPSolution)::Nothing

    # Parameters
    algo = sol.algo
    veh = mdl.vehicle
    spread = 0.4
    stem_colors = [Red, Green, Blue, DarkBlue] #noerr
    marker_darken_factor = 0.3
    padx=0.05

    # Plotting data
    dt_τ = sol.td
    tf = sol.p[veh.id_t]
    dt_t = dt_τ*tf
    dt_res = length(dt_τ)
    Δt = tf/(dt_res-1)
    t_spread = Δt*spread/2

    T = sol.ud[veh.id_T, :]
    M = sol.ud[veh.id_M, :]

    dirs = ["\$+x\$", "\$+y\$", "\$+z\$"]
    data = [Dict(:u=>T,
                 :ylabel=>"Force impulse [N\$\\cdot\$s]",
                 :legend=>(i)->@sprintf("Along %s", dirs[i])),
            Dict(:u=>M,
                 :ylabel=>"Torque impulse [N\$\\cdot\$m\$\\cdot\$s]",
                 :legend=>(i)->@sprintf("Along %s", dirs[i]))]

    fig = create_figure((8, 6))
    gspec = fig.add_gridspec(ncols=1, nrows=2)

    axes = []

    for i_plt = 1:length(data)

        # Data
        u = data[i_plt][:u]
        num_inputs = size(u, 1)

        ax = setup_axis!(gspec[i_plt],
                         ylabel=data[i_plt][:ylabel])
        push!(axes, ax)

        if i_plt==1
            ax.tick_params(axis="x", which="both", bottom=false, top=false,
                           labelbottom=false)
        else
            ax.set_xlabel("Time [s]")
        end

        # Plot the stems
        for k = 1:dt_res
            for i = 1:num_inputs

                xloc = dt_t[k]-t_spread+(i-1)/(num_inputs-1)*2*t_spread
                uik = u[i, k]
                clr = stem_colors[i]
                lbl = (k==1) ? data[i_plt][:legend](i) : nothing

                # Stem line
                ax.plot([xloc, xloc], [0, uik],
                        linewidth=2,
                        color=clr,
                        solid_capstyle="round",
                        clip_on=false,
                        zorder=20)

                # Stem tip/"flower"
                darker_clr = darken_color(clr, marker_darken_factor)
                ax.plot(xloc, uik,
                        linestyle="none",
                        marker="o",
                        markersize=4,
                        markeredgecolor="white",
                        markeredgewidth=0.2,
                        markerfacecolor=darker_clr,
                        clip_on=false,
                        zorder=30)

                # "Fake" step tip just for the legend
                ax.plot([], [],
                        linestyle="none",
                        marker="o",
                        markersize=4.2,
                        markerfacecolor=clr,
                        markeredgewidth=0,
                        zorder=-1,
                        label=lbl)

            end
        end

        # Set axis limit
        x_rng = (tf+t_spread)
        x_pad = padx*x_rng/2
        xmin = -t_spread-x_pad
        xmax = x_rng+x_pad
        ax.set_xlim(xmin, xmax)

        # Plot "zero" baseline
        ax.plot([xmin, xmax], [0, 0],
                color=DarkBlue, #noerr
                linewidth=0.5,
                solid_capstyle="round",
                zorder=10)

        # Legend
        leg = ax.legend(framealpha=0.8,
                        fontsize=8,
                        loc="upper left")
        leg.set_zorder(100)

    end

    fig.align_ylabels(axes)

    save_figure("rendezvous_inputs", algo)

    return nothing
end # function
