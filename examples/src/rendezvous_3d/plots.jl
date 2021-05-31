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

#nolint: create_figure, save_figure, generate_colormap, squeeze
#nolint: setup_axis!, darken_color
#nolint: Red, DarkBlue, Yellow, Green, Blue
#nolint: or, set_axis_equal, rgb2pyplot
#nolint: SCPSolution
#nolint: sample

LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("parameters.jl")
    include("definition.jl")
end

using PyPlot
using Colors

using Solvers #noerr

# ..:: Globals ::..

const fig_opts = Dict(
    "font.family"=>"serif",
    "text.latex.preamble"=>
        string("\\usepackage{newtxtext}",
               "\\usepackage{newtxmath}",
               "\\usepackage{siunitx}"))

const gray = rgb2pyplot(weighted_color_mean(
    0.1, colorant"black", colorant"white"))

const red = rgb2pyplot(weighted_color_mean(
    0.2, parse(RGB, Red), colorant"white"))

const time_mark_clr = rgb2pyplot(weighted_color_mean(
    0.5, colorant"black", colorant"white"))

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
    traj = mdl.traj
    axis_inf_scale = 1e4
    pad_x = 0.1
    pad_y = 0.1
    pad_label_abs = 10
    equal_ar = true # Equal aspect ratio
    input_scale = 0.2

    # Plotting data
    ct_res = 1000
    dt_res = length(sol.td)
    ct_τ = RealVector(LinRange(0, 1, ct_res))
    dt_pos = sol.xd[veh.id_r, :]
    ct_pos = hcat([sample(sol.xc, τ)[veh.id_r] for τ in ct_τ]...) #noerr
    ct_vel = hcat([sample(sol.xc, τ)[veh.id_v] for τ in ct_τ]...) #noerr
    ct_speed = squeeze(mapslices(norm, ct_vel, dims=(1)))
    n_rcs = length(veh.id_rcs)
    N = size(sol.ud, 2)
    dt_thrust = zeros(3, N)
    dir_rcs = [veh.csm.f_rcs[veh.csm.rcs_select[i]] for i=1:n_rcs]
    for k = 1:size(dt_thrust, 2)
        q = Quaternion(sol.xd[veh.id_q, k])
        dir_rcs_iner = [rotate(dir_rcs[i], q) for i=1:n_rcs] #noerr
        dt_thrust[:, k] = sum(sol.ud[veh.id_rcs[i], k]*dir_rcs_iner[i]
                              for i=1:n_rcs)
    end

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

    fig = create_figure((6, 10), options=fig_opts)

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
                       :y=>pad_y*(bbox[:y][:max]-bbox[:y][:min]))

        # Plot axes "guidelines"
        origin = [0; 0]#mdl.traj.rf[prj]
        xtip = [1; 0]
        ytip = [0; 1]
        ax.plot([origin[1], origin[1]+xtip[1]*axis_inf_scale],
                [origin[2], origin[2]+xtip[2]*axis_inf_scale],
                color=DarkBlue, #noerr
                linewidth=0.5,
                solid_capstyle="round")
        ax.plot([origin[1], origin[1]+ytip[1]*axis_inf_scale],
                [origin[2], origin[2]+ytip[2]*axis_inf_scale],
                color=DarkBlue, #noerr
                linewidth=0.5,
                solid_capstyle="round")

        # Plot the continuous-time trajectory
        line_segs = Vector{RealMatrix}(undef, 0)
        line_clrs = Vector{NTuple{4, RealValue}}(undef, 0)
        overlap = 3
        for k=1:ct_res-overlap
            push!(line_segs, ct_pos_2d[:, k:k+overlap]')
            push!(line_clrs, v_cmap.to_rgba(ct_speed[k]))
        end
        trajectory = PyPlot.matplotlib.collections.LineCollection(
            line_segs, zorder=10,
            colors = line_clrs,
            linewidths=3)
        ax.add_collection(trajectory)

        # Plot the discrete-time trajectory
        ax.scatter(dt_pos_2d[1, :], dt_pos_2d[2, :],
                   marker="o",
                   c=DarkBlue, #noerr
                   s=5,
                   edgecolors="white",
                   linewidths=0.2,
                   zorder=20)

        # Axis limits
        pad_value = max(padding[:x], padding[:y])
        xmin = bbox[:x][:min]-pad_value
        xmax = bbox[:x][:max]+pad_value
        ymin = bbox[:y][:min]-pad_value
        ymax = bbox[:y][:max]+pad_value

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
            if (any(dt_pos_2d[2, :].>y_rng[2]) ||
                any(dt_pos_2d[2, :].<y_rng[1]))
                # The data does not fit, leave xmax unconstrained instead
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
        thrusts = PyPlot.matplotlib.collections.LineCollection(
            thrust_segs, zorder=15,
            colors=Red, #noerr
            linewidths=1.5)
        thrusts.set_capstyle("round")
        ax.add_collection(thrusts)

        # Label the LVLH axes
        ax.annotate(@sprintf("\$\\hat %s_{\\mathcal L}\$", ax_x_name),
                    xy=(x_rng[2], origin[2]),
                    xytext=(-pad_label_abs, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    bbox=Dict(:pad=>2, :fc=>"white", :ec=>"none"))
        ax.annotate(@sprintf("\$\\hat %s_{\\mathcal L}\$", ax_y_name),
                    xy=(origin[1], y_rng[2]),
                    xytext=(0, pad_label_abs),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    bbox=Dict(:pad=>2, :fc=>"white", :ec=>"none"))

        # Plume cone
        if 1 in prj
            θ_sweep_sph = LinRange(0, 2*pi, 100)
            sph_perim = hcat(map(θ->traj.r_plume*[cos(θ); sin(θ)],
                                 θ_sweep_sph)...)
            sph_x = sph_perim[1, :]
            sph_y = sph_perim[2, :]

            ax.plot(sph_x, sph_y,
                    color=Red,
                    linewidth=0.6,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=4)
        end

        # Approach cone
        if 1 in prj
            # Side-view
            θ_sweep = LinRange(-traj.θ_appch, traj.θ_appch, 100)
            cone_perim = hcat(map(θ->traj.r_appch*[cos(θ); sin(θ)],
                                  θ_sweep)...)
            cone_x = [0; cone_perim[1, :]; 0]
            cone_y = [0; cone_perim[2, :]; 0]

            # Draw the approach sphere
            θ_sweep_sph = LinRange(traj.θ_appch, 2*pi-traj.θ_appch, 100)
            sph_perim = hcat(map(θ->traj.r_appch*[cos(θ); sin(θ)],
                                 θ_sweep_sph)...)
            sph_x = sph_perim[1, :]
            sph_y = sph_perim[2, :]

            ax.plot(sph_x, sph_y,
                    color=DarkBlue,
                    linewidth=0.6,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=4)
        else
            # Front-on view
            θ_sweep = LinRange(0, 2*pi, 100)
            r_intersect = traj.r_appch*sin(traj.θ_appch)
            cone_perim = hcat(map(θ->r_intersect*[cos(θ); sin(θ)],
                                  θ_sweep)...)
            cone_x = cone_perim[1, :]
            cone_y = cone_perim[2, :]
        end
        ax.plot(cone_x, cone_y,
                color=Green,
                linestyle="--",
                linewidth=1.5,
                solid_capstyle="round",
                solid_joinstyle="round",
                dash_capstyle="round",
                dash_joinstyle="round",
                dashes=(3, 3),
                zorder=5)

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

    save_figure("rendezvous_3d_trajectory_2d.pdf", algo, tight_layout=false)

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
    traj = mdl.traj
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

    # Find approach and plume sphere entry times
    k_appch = findfirst(k->norm(ct_r[:, k])<=traj.r_appch, 1:ct_res)
    k_plume = findfirst(k->norm(ct_r[:, k])<=traj.r_plume, 1:ct_res)
    t_appch = ct_t[k_appch]
    t_plume = ct_t[k_plume]

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
    fig = create_figure((10, 10), options=fig_opts)
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

        # Plot reflines for the approach and plume sphere entry times
        label_text = Dict(t_appch=>"Approach", t_plume=>"Plume")
        ax_ylim = ax.get_ylim()
        for t in [t_appch, t_plume]
            ax.axvline(x=t,
                       color=time_mark_clr,
                       linestyle="--",
                       linewidth=0.7,
                       dash_joinstyle="round",
                       dash_capstyle="round",
                       dashes=(3, 3),
                       zorder=5)

            ax.annotate(label_text[t] ,
                        color=time_mark_clr,
                        xy=(t, ax_ylim[2]),
                        xytext=(0, -10),#0.1*(ax_ylim[2]-ax_ylim[1])),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        rotation=90,
                        zorder=5,
                        bbox=Dict(:pad=>1, :fc=>"white", :ec=>"none"))
        end

        # Make an x tick for the final time
        ax_xticks = ax.get_xticks()
        ax_xlim = ax.get_xlim()
        if ax_xticks[end]!=tf
            push!(ax_xticks, tf)
        end
        ax.set_xticks(ax_xticks)
        ax.set_xlim(ax_xlim)

    end

    fig.align_ylabels(ax_list[1:3])
    fig.align_ylabels(ax2_list[1:3])
    fig.align_ylabels(ax_list[4:6])
    fig.align_ylabels(ax2_list[4:6])

    save_figure("rendezvous_3d_timeseries.pdf", algo)

    return nothing
end # function

"""
    plot_control(mdl, sol)

Plot the control inputs versus time.

# Arguments
- `mdl`: the problem description object.
- `sol`: the problem solution.
- `history`: SCP iteration data history.
"""
function plot_inputs(mdl::RendezvousProblem,
                     sol::SCPSolution, #noerr
                     history::SCPHistory)::Nothing #noerr

    # Parameters
    interm_sol = [spbm.sol for spbm in history.subproblems]
    num_iter = length(interm_sol)
    algo = sol.algo
    veh = mdl.vehicle
    traj = mdl.traj
    spread = 0.4
    stem_colors = [Red, Green, Blue, DarkBlue] #noerr
    marker_darken_factor = 0.3
    padx=0.05
    polar_resol = 1000

    # Plotting data
    dt_τ = sol.td
    tf = sol.p[veh.id_t]
    dt_t = dt_τ*tf
    dt_res = length(dt_τ)
    Δt = tf/(dt_res-1)
    t_spread = Δt*spread/2

    # Find approach and plume sphere entry times
    ct_res = 1000
    ct_τ = RealVector(LinRange(0, 1, ct_res))
    ct_t = ct_τ*tf
    ct_pos = hcat([sample(sol.xc, τ)[veh.id_r] for τ in ct_τ]...)
    k_appch = findfirst(k->norm(ct_pos[:, k])<=traj.r_appch, 1:ct_res)
    k_plume = findfirst(k->norm(ct_pos[:, k])<=traj.r_plume, 1:ct_res)
    t_appch = ct_t[k_appch]
    t_plume = ct_t[k_plume]

    # Get RCS controls solution history
    f_quad = Dict[]
    f_quad_ref = Dict[]
    hom_val = Real[]
    for i = 1:num_iter

        f = interm_sol[i].ud[veh.id_rcs, :]
        f_ref = interm_sol[i].ud[veh.id_rcs_ref, :]

        push!(f_quad, Dict())
        push!(f_quad_ref, Dict())
        push!(hom_val, interm_sol[i].bay[:hom])

        for quad in (:A, :B, :C, :D)
            _f_quad = RealVector[]
            _f_quad_ref = RealVector[]
            for thruster in (:pf, :pa, :rf, :ra)
                push!(_f_quad,
                      f[veh.csm.rcs_select[quad, thruster], :])
                push!(_f_quad_ref,
                      f_ref[veh.csm.rcs_select[quad, thruster], :])
            end
            f_quad[i][quad] = hcat(_f_quad...)'
            f_quad_ref[i][quad] = hcat(_f_quad_ref...)'
        end
    end

    dirs = ["\$+\\hat x_{\\mathcal B}\$",
            "\$-\\hat x_{\\mathcal B}\$",
            "\$+\\hat y_{\\mathcal B}\$ (A)",
            "\$-\\hat y_{\\mathcal B}\$ (A)"]
    thruster_label = (i) -> @sprintf("Thruster %s", dirs[i])
    data = [Dict(:u=>(i)->f_quad[i][:A],
                 :u_ref=>(i)->f_quad_ref[i][:A],
                 :ylabel=>"Quad A impulse [\\si{\\newton\\second}]",
                 :legend=>thruster_label),
            Dict(:u=>(i)->f_quad[i][:B],
                 :u_ref=>(i)->f_quad_ref[i][:B],
                 :ylabel=>"Quad B impulse [\\si{\\newton\\second}]",
                 :legend=>thruster_label),
            Dict(:u=>(i)->f_quad[i][:C],
                 :u_ref=>(i)->f_quad_ref[i][:C],
                 :ylabel=>"Quad C impulse [\\si{\\newton\\second}]",
                 :legend=>thruster_label),
            Dict(:u=>(i)->f_quad[i][:D],
                 :u_ref=>(i)->f_quad_ref[i][:D],
                 :ylabel=>"Quad D impulse [\\si{\\newton\\second}]",
                 :legend=>thruster_label)]

    fig = create_figure((10, 18), options=fig_opts)
    gspec = fig.add_gridspec(ncols=2, nrows=6,
                             width_ratios=[0.7, 0.3])

    axes = []

    for i_plt = 1:length(data)

        # Data
        u = data[i_plt][:u](num_iter)
        ur = data[i_plt][:u_ref](num_iter)
        num_inputs = size(u, 1)
        fr_rng = LinRange(0, veh.csm.imp_max, polar_resol)
        or_mib_normalize = veh.csm.imp_max-veh.csm.imp_min
        above_mib = (fr)->fr-veh.csm.imp_min
        f_polar = (hom) -> map(fr_rng) do fr
            or([above_mib(fr)],
               κ=hom, match=or_mib_normalize,
               normalize=or_mib_normalize)*fr
        end

        # >> Draw the timeseries plot <<

        ax = setup_axis!(gspec[i_plt, 1],
                         ylabel=data[i_plt][:ylabel])
        push!(axes, ax)

        if i_plt<length(data)
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
                        linewidth=1.5,
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

                # "Fake" stem tip just for the legend
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
        ax.set_ylim(nothing, veh.csm.imp_max-ax.get_ylim()[1])
        ylim_timeseries = ax.get_ylim()

        # Plot "zero" baseline
        ax.plot([xmin, xmax], [0, 0],
                color=DarkBlue, #noerr
                linewidth=0.5,
                solid_capstyle="round",
                zorder=10)

        # Plot reflines for the approach and plume sphere entry times
        label_text = Dict(t_appch=>"Approach", t_plume=>"Plume")
        for t in [t_appch, t_plume]
            ax.axvline(x=t,
                       color=time_mark_clr,
                       linestyle="--",
                       linewidth=0.7,
                       dash_joinstyle="round",
                       dash_capstyle="round",
                       dashes=(3, 3),
                       zorder=10)

            ax.annotate(label_text[t] ,
                        color=time_mark_clr,
                        xy=(t, ylim_timeseries[2]),
                        xytext=(0, -10),
                        textcoords="offset points",
                        ha="center",
                        va="top",
                        rotation=90,
                        zorder=10,
                        bbox=Dict(:pad=>1, :fc=>"white", :ec=>"none"))
        end

        # Legend
        if i_plt==1
            leg = ax.legend(framealpha=0.8,
                            fontsize=8,
                            loc="upper center",
                            ncol=4,
                            bbox_to_anchor=(0.5, 1.2))
            leg.set_zorder(100)
        end

        # Make an x tick for the final time
        ax_xticks = ax.get_xticks()
        ax_xlim = ax.get_xlim()
        if ax_xticks[end]!=tf
            push!(ax_xticks, tf)
        end
        ax.set_xticks(ax_xticks)
        ax.set_xlim(ax_xlim)

        # >> Draw the deadband polar plot <<

        ax = setup_axis!(gspec[i_plt, 2];
                         axis="square")

        ax.tick_params(axis="y", which="both", left=false, right=false,
                       labelleft=false)

        if i_plt<length(data)
            ax.tick_params(axis="x", which="both", bottom=false, top=false,
                           labelbottom=false)
        else
            ax.set_xlabel("Impulse reference [\\si{\\newton\\second}]")
        end

        # Continuous polar without deadband
        ax.plot(fr_rng, fr_rng,
                color=DarkBlue,
                linewidth=1,
                linestyle="--",
                solid_capstyle="round",
                dash_capstyle="round",
                dashes=(3, 3),
                zorder=100)

        # Continuous polar with deadband
        ax.plot(fr_rng, f_polar(hom_val[end]),
                color=Yellow,
                linewidth=2,
                solid_capstyle="round",
                zorder=100,
                clip_on=true)

        # The discrete-time (ref, actual) inputs
        for i = 1:num_inputs
            clr = stem_colors[i]
            darker_clr = darken_color(clr, marker_darken_factor)

            ax.plot(ur[i, :], u[i, :],
                    linestyle="none",
                    marker="o",
                    markersize=4,
                    markeredgecolor="white",
                    markeredgewidth=0.2,
                    markerfacecolor=darker_clr,
                    zorder=100)
        end

        ax.set_xlim(ylim_timeseries)
        ax.set_ylim(ylim_timeseries)

    end

    plt.subplots_adjust(wspace=0.02, hspace=0.1)
    fig.align_ylabels(axes)

    save_figure("rendezvous_3d_inputs.pdf", algo, tight_layout=false)

    return nothing
end # function

"""
    plot_cost_evolution(mdl, sol)

Plot how the cost versus algorithm iterations.

# Arguments
- `mdl`: the problem description object.
- `history`: SCP iteration data history.
"""
function plot_cost_evolution(mdl::RendezvousProblem,
                             history::SCPHistory)::Nothing

    # Parameters
    algo = history.subproblems[1].algo
    nxticks = 8
    β = mdl.traj.β*100
    β_lower = -1e-1
    ylabel_size = 12

    # Values
    Niter = length(history.subproblems)
    iters = collect(1:Niter)
    J = map(spbm->spbm.sol.J_aug, history.subproblems)
    dJ = map(i->(J[i-1]-J[i])/abs(J[i-1]), 2:Niter)*100
    hom = map(spbm->spbm.sol.bay[:hom], history.subproblems)
    hom_updated = map(spbm->spbm.sol.bay[:hom_updated],
                      history.subproblems)

    fig = create_figure((7, 6), options=fig_opts)
    axes = []

    # >> Plot the cost evolution <<

    ax = setup_axis!(311,
                     ylabel="Cost function value, \$J_{\\ell}\$")
    push!(axes, ax)
    ax.tick_params(axis="x", which="both", bottom=false, top=false,
                   labelbottom=false)

    ax.plot(iters, J,
            color=gray,
            marker="o",
            markersize=5,
            markerfacecolor=DarkBlue,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=20,
            solid_capstyle="round",
            solid_joinstyle="round",
            clip_on=false)

    for iter = 1:Niter
        if hom_updated[iter]
            ax.plot(iter, J[iter],
                    marker="o",
                    markersize=5,
                    markerfacecolor=Red,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    zorder=25,
                    clip_on=false)
        end
    end

    # X-ticks (iterations)
    xticks = map(x->round(Int, x), range(1, Niter, length=nxticks))
    ax.set_xticks(xticks)

    ax.set_xlim(0, Niter+1)

    # >> Plot the relative cost change per iteration <<

    ax = setup_axis!(312,
                     ylabel="Cost decrease, "*
                         "\$\\frac{J_{\\ell-1}-J_{\\ell}}{|J_{\\ell-1}|}\$"*
                         " [\\%]")
    push!(axes, ax)
    ax.tick_params(axis="x", which="both", bottom=false, top=false,
                   labelbottom=false)

    ax.plot(iters[2:end], dJ,
            color=gray,
            marker="o",
            markersize=5,
            markerfacecolor=DarkBlue,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=20,
            solid_capstyle="round",
            solid_joinstyle="round",
            clip_on=false)

    # Trigger boudns
    for i=1:2
        ax.axhline(y=(i==1) ? β : β_lower,
                   zorder=15,
                   color=Red,
                   linestyle="--",
                   linewidth=1,
                   dash_joinstyle="round",
                   dash_capstyle="round",
                   dashes=(3, 3),
                   clip_on=true)

        ax.annotate((i==1) ? "\$\\beta_{\\mathrm{trig}}\$" :
            "\$\\beta_{\\mathrm{worse}}\$",
                    color=Red,
                    xy=(1, (i==1) ? β : β_lower),
                    xytext=(0, ((i==1) ? 1 : -1)*2),
                    textcoords="offset points",
                    ha="center",
                    va=(i==1) ? "bottom" : "top",
                    zorder=30,
                    bbox=Dict(:pad=>1, :fc=>"white", :ec=>"none"))
    end

    for iter = 2:Niter
        if hom_updated[iter]
            ax.plot(iter, dJ[iter-1],
                    marker="o",
                    markersize=5,
                    markerfacecolor=Red,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    zorder=25,
                    clip_on=false)
        end
    end

    # X-ticks (iterations)
    xticks = map(x->round(Int, x), range(1, Niter, length=nxticks))
    ax.set_xticks(xticks)
    ax.set_xlim(0, Niter+1)

    # Y-ticks
    yticks = ax.get_yticks()
    ylims = ax.get_ylim()
    if yticks[end]!=maximum(dJ)
        yticks[end] = maximum(dJ)
    end
    ax.set_yticks(yticks)
    ax.set_ylim(ylims)

    # >> Homotopy value evolution <<

    ax = setup_axis!(313,
                     xlabel="Iteration number, \$\\ell\$",
                     ylabel="Homotopy parameter, \$\\kappa\$")
    push!(axes, ax)
    ax.set_yscale("log")

    ax.plot(iters, hom,
            color=gray,
            marker="o",
            markersize=5,
            markerfacecolor=DarkBlue,
            markeredgecolor="white",
            markeredgewidth=1,
            zorder=20,
            solid_capstyle="round",
            solid_joinstyle="round",
            clip_on=false)

    for iter = 1:Niter
        if hom_updated[iter]
            ax.plot(iter, hom[iter],
                    marker="o",
                    markersize=5,
                    markerfacecolor=Red,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    zorder=25,
                    clip_on=false)
        end
    end

    for iter = 1:Niter
        if hom_updated[iter]
            for axi in axes
                axi.axvline(
                    x=iter,
                    color=Red,
                    linestyle="--",
                    linewidth=0.7,
                    dash_joinstyle="round",
                    dash_capstyle="round",
                    dashes=(3, 3),
                    zorder=10)
            end
        end
    end

    # X-ticks (iterations)
    xticks = map(x->round(Int, x), range(1, Niter, length=nxticks))
    ax.set_xticks(xticks)

    ax.set_xlim(0, Niter+1)

    # >> Finalize figure and save <<

    map(ax->ax.yaxis.label.set_size(ylabel_size), axes)
    fig.align_ylabels(axes)

    save_figure("rendezvous_3d_cost.pdf", algo)

    return nothing
end # function

"""
    plot_homotopy_threshold_sweep(mdl, betas, sols)

Plot the effect of the homotopy update threshold on the number of iterations
and optimal cost.

# Arguments
- `mdl`: the problem description object.
- `betas`: the array of beta values tested.
- `sols`: list of solutions for each beta value.
"""
function plot_homotopy_threshold_sweep(
    mdl::RendezvousProblem, #nowarn
    betas::RealVector,
    sols::Vector{SCPSolution})::Nothing

    # Parameters
    algo = sols[1].algo

    # Values
    impulses = sol->sol.ud[mdl.vehicle.id_rcs, :]
    J = map(sol->sol.cost, sols) #noinfo
    fuel = map(sol->fuel_consumption(mdl, impulses(sol)), sols)
    iters = map(sol->sol.iterations, sols)

    create_figure((7, 3), options=fig_opts)

    # >> Plot the effect on cost <<
    axJ = setup_axis!(111,
                      xlabel="Homotopy update tolerance "*
                          "\$\\beta_{\\mathrm{trig}}\$")
    axJ.grid(false)
    axJ_clr = DarkBlue
    axJ.set_ylabel("Fuel consumption [\\si{\\kilo\\gram}]",
                   color=axJ_clr)
    axJ.tick_params(axis="y", colors=axJ_clr)
    axJ.spines["right"].set_edgecolor(axJ_clr)

    axJ.plot(betas, fuel,
             color=gray,
             linewidth=2,
             marker="o",
             markersize=5,
             markerfacecolor=DarkBlue,
             markeredgecolor="white",
             markeredgewidth=1,
             zorder=100,
             solid_capstyle="round",
             solid_joinstyle="round",
             clip_on=false)

    xticks = axJ.get_xticks()
    xlims = axJ.get_xlim()
    xticks = filter(x->x>=0, xticks)
    xticks[1] = betas[1]
    xticks[end] = betas[end]
    axJ.set_xticks(xticks)
    axJ.set_xlim(xlims)

    yticks = axJ.get_yticks()
    ylims = axJ.get_ylim()
    if yticks[1]!=minimum(fuel)
        pushfirst!(yticks, round(minimum(fuel), digits=2))
    end
    if yticks[end]!=maximum(fuel)
        push!(yticks, round(maximum(fuel), digits=2))
    end
    axJ.set_yticks(yticks)
    axJ.set_ylim(ylims)

    # >> Plot the effect on the number of iterations <<
    axN = axJ.twinx()
    axN.grid(linewidth=0.3, alpha=0.5)
    axN_clr = Red
    outline_w = 1.5
    axN.set_ylabel("Number of PTR iterations", color=axN_clr)
    axN.tick_params(axis="y", colors=axN_clr)
    axN.spines["right"].set_edgecolor(axN_clr)
    axN.tick_params(axis="x", which="both", bottom=false, top=false,
                    labelbottom=false)

    axN.plot(betas, iters,
             color=red,
             linewidth=2,
             marker="o",
             markersize=5,
             markerfacecolor=Red,
             markeredgewidth=0,
             zorder=20,
             solid_capstyle="round",
             solid_joinstyle="round",
             clip_on=false)

    axN.plot(betas, iters,
             color="white",
             linewidth=2+outline_w,
             marker="o",
             markersize=5,
             markerfacecolor="white",
             markeredgecolor="white",
             markeredgewidth=outline_w,
             zorder=19,
             solid_capstyle="round",
             solid_joinstyle="round",
             clip_on=false)

    yticks = axN.get_yticks()
    nyticks = 5
    yticks = map(y->round(Int, y),
                 range(minimum(iters), maximum(iters), length=nyticks))
    axN.set_yticks(yticks)

    save_figure("rendezvous_3d_homotopy_threshold.pdf", algo)

    return nothing
end # function
