#= Rocket landing plots.

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

using PyPlot
using Colors
using Printf

"""
    plot_thrust(rocket, sol)

Plot the optimal thrust time history.

# Arguments
- `rocket`: the rocket definition.
- `sol`: the discrete-time solution from the optimizer.
- `sim`: the continuous-time post-processed simulation.
"""
function plot_thrust(rocket::Rocket, sol::Solution, sim::Solution)::Nothing

    # Parameters
    tf = sol.t[end]
    scale = 1e-3 # N->kN
    min_thrust = rocket.ρ_min*scale
    max_thrust = rocket.ρ_max*scale
    t_dt = sol.t
    t_ct = sim.t
    N = length(t_dt)
    N_sim = length(t_ct)
    T_dt = sol.T*scale
    ξ_dt = sol.ξ
    m_dt = sol.m
    T_ct = sim.T*scale
    T_dt_nrm = squeeze(mapslices(norm, T_dt, dims=1))
    T_ct_nrm = squeeze(mapslices(norm, T_ct, dims=1))
    T_dt_z = T_dt[3, :]
    T_ct_z = T_ct[3, :]
    σ_dt = m_dt[1:end-1].*ξ_dt*scale
    max_thrust_z = squeeze(mapslices(
        T->sqrt((max_thrust)^2-T[1]^2-T[2]^2),
        T_dt, dims=1))
    min_thrust_z = squeeze(mapslices(
        T->sqrt(max(0, (min_thrust)^2-T[1]^2-T[2]^2)),
        T_dt, dims=1))
    hover_line = sol.m*norm(rocket.g)*scale

    # ..:: Thrust magnitude plot ::..

    fig = create_figure((4, 4))

    ax = setup_axis!(111,
                     xlabel="Time [s]",
                     ylabel="Thrust [kN]",
                     tight="both")

    # Thrust bounds
    plot_timeseries_bound!(ax, 0.0, tf, min_thrust, -min_thrust; lw=1.25)
    plot_timeseries_bound!(ax, 0.0, tf, max_thrust, 16-max_thrust; lw=1.25)

    # Continuous-time thrust
    ax.plot(t_ct, T_ct_nrm,
            linewidth=1.5,
            color=DarkBlue,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=20,
            clip_on=false,
            label="\$\\|T_c(t)\\|_2\$")

    # Discrete-time thrust
    ax.plot(t_dt[1:end-1], T_dt_nrm,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor=Blue,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=20,
            clip_on=false,
            label="\$\\|T_{c,k}\\|_2\$")

    # Discrete-time slack variable
    ax.plot(t_dt[1:end-1], σ_dt,
            linestyle="none",
            marker="h",
            markersize=2,
            markerfacecolor=Yellow,
            markeredgecolor=Blue,
            markeredgewidth=0.2,
            zorder=20,
            clip_on=false,
            label="\$\\sigma_k\$")

    ax.set_xlim(0, round(tf, digits=3))
    ax.set_xticks(vcat(ax.get_xticks(), round(Int, tf)))
    ax.set_xlim(0, round(tf, digits=3))

    leg = ax.legend(framealpha=0.8, fontsize=8,
                    loc="lower left")
    leg.set_zorder(200)

    save_figure("rocket_landing_thrust_mag.pdf", "lcvx")

    # ..:: Thrust z-component plot ::..

    fig = create_figure((4, 4))

    ax = setup_axis!(111,
                     xlabel="Time [s]",
                     ylabel="Thrust along \$\\hat e_z\$ [kN]",
                     tight="both")

    # Thrust bounds
    plot_timeseries_bound!(ax, 0.0, tf, min_thrust_z, 0; lw=1.25, abs=true)
    plot_timeseries_bound!(ax, 0.0, tf, max_thrust_z, 16; lw=1.25, abs=true)

    # Hover thrust line
    ax.plot(t_dt, hover_line,
            linewidth=1.5,
            color=Green,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=15,
            clip_on=false,
            label="\$\\hat e_z^{\\scriptscriptstyle\\mathsf{T}}gm_k\$ (hover)")

    # Continuous-time thrust
    ax.plot(t_ct, T_ct_z,
            linewidth=1.5,
            color=DarkBlue,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=20,
            clip_on=false,
            label="\$\\hat e_z^{\\scriptscriptstyle\\mathsf{T}}T_c(t)\$")

    # Discrete-time thrust
    ax.plot(t_dt[1:end-1], T_dt_z,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor=Blue,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=20,
            clip_on=false,
            label="\$\\hat e_z^{\\scriptscriptstyle\\mathsf{T}}T_{c,k}\$")

    ax.set_xlim(0, round(tf, digits=3))
    ax.set_xticks(vcat(ax.get_xticks(), round(Int, tf)))
    ax.set_xlim(0, round(tf, digits=3))

    leg = ax.legend(framealpha=0.8, fontsize=8,
                    loc="lower left")
    leg.set_zorder(200)

    save_figure("rocket_landing_thrust_z.pdf", "lcvx")

    return nothing
end

"""
    plot_mass(rocket, sol)

Plot the optimal mass time history.

# Arguments
- `rocket`: the rocket definition.
- `sol`: the discrete-time solution from the optimizer.
- `sim`: the continuous-time post-processed simulation.
"""
function plot_mass(rocket::Rocket, sol::Solution, sim::Solution)::Nothing

    # Parameters
    tf = sol.t[end]
    scale = 1e-3 # kg->t
    min_mass = rocket.m_dry*scale
    max_mass = rocket.m_wet*scale
    t_dt = sol.t
    t_ct = sim.t
    N = length(t_dt)
    N_sim = length(t_ct)
    m_dt = sol.m*scale
    m_ct = sim.m*scale

    # ..:: Mass plot ::..

    fig = create_figure((4, 4))

    ax = setup_axis!(111,
                     xlabel="Time [s]",
                     ylabel="Mass [t]",
                     tight="both")

    # Mass bounds
    plot_timeseries_bound!(ax, 0.0, tf, min_mass, 1.45; lw=1.25, abs=true)
    plot_timeseries_bound!(ax, 0.0, tf, max_mass, 1.95; lw=1.25, abs=true)

    # Continuous-time thrust
    ax.plot(t_ct, m_ct,
            linewidth=1.5,
            color=DarkBlue,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=20,
            clip_on=false,
            label="\$m(t)\$")

    # Discrete-time thrust
    ax.plot(t_dt, m_dt,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor=Blue,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=20,
            clip_on=false,
            label="\$m_k\$")

    leg = ax.legend(framealpha=0.8, fontsize=8,
                    loc="lower left")
    leg.set_zorder(200)

    ax.set_xlim(0, round(tf, digits=3))
    ax.set_xticks(vcat(ax.get_xticks(), round(Int, tf)))
    ax.set_xlim(0, round(tf, digits=3))

    save_figure("rocket_landing_mass.pdf", "lcvx")

    return nothing
end

"""
    plot_velocity(rocket, sol)

Plot the velocity norm time history.

# Arguments
- `rocket`: the rocket definition.
- `sol`: the discrete-time solution from the optimizer.
- `sim`: the continuous-time post-processed simulation.
"""
function plot_velocity(rocket::Rocket, sol::Solution,
                       sim::Solution)::Nothing

    # Parameters
    tf = sol.t[end]
    scale = 3600/1000 # m/s->km/h
    v_max = rocket.v_max*scale
    t_dt = sol.t
    t_ct = sim.t
    N = length(t_dt)
    N_sim = length(t_ct)
    v_dt = sol.v
    v_ct = sim.v
    v_nrm_dt = squeeze(mapslices(norm, v_dt, dims=1))*scale
    v_nrm_ct = squeeze(mapslices(norm, v_ct, dims=1))*scale

    # ..:: Velocity plot ::..

    fig = create_figure((4, 4))

    ax = setup_axis!(111,
                     xlabel="Time [s]",
                     ylabel="Speed \$\\|v(t)\\|_2\$ [km/h]",
                     tight="both")

    # Mass bounds
    plot_timeseries_bound!(ax, 0.0, tf, v_max, 600; lw=1.25, abs=true)

    # Continuous-time thrust
    ax.plot(t_ct, v_nrm_ct,
            linewidth=1.5,
            color=DarkBlue,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=20,
            clip_on=false,
            label="\$\\|v(t)\\|_2\$")

    # Discrete-time thrust
    ax.plot(t_dt, v_nrm_dt,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor=Blue,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=20,
            clip_on=false,
            label="\$\\|v_k\\|_2\$")

    ax.set_ylim(0, nothing)

    leg = ax.legend(framealpha=0.8, fontsize=8,
                    loc="lower left")
    leg.set_zorder(200)

    ax.set_xlim(0, round(tf, digits=3))
    ax.set_xticks(vcat(ax.get_xticks(), round(Int, tf)))
    ax.set_xlim(0, round(tf, digits=3))

    save_figure("rocket_landing_velocity.pdf", "lcvx")

    return nothing
end

"""
    plot_pointing_angle(rocket, sol)

Plot the thrust pointing angle time history.

# Arguments
- `rocket`: the rocket definition.
- `sol`: the discrete-time solution from the optimizer.
- `sim`: the continuous-time post-processed simulation.
"""
function plot_pointing_angle(rocket::Rocket, sol::Solution,
                             sim::Solution)::Nothing

    # Parameters
    tf = sol.t[end]
    scale = 180/pi # rad->deg
    θ_max = rocket.γ_p*scale
    t_dt = sol.t
    t_ct = sim.t
    N = length(t_dt)
    N_sim = length(t_ct)
    T_dt = sol.T
    T_ct = sim.T
    θ_dt = squeeze(mapslices(T->acos(T[3]/norm(T)), T_dt, dims=1))*scale
    θ_ct = squeeze(mapslices(T->acos(T[3]/norm(T)), T_ct, dims=1))*scale

    # ..:: Pointing angle plot ::..

    fig = create_figure((4, 4))

    ax = setup_axis!(111,
                     xlabel="Time [s]",
                     ylabel="Pointing angle [\$^{\\circ}\$]",
                     tight="both")

    # Mass bounds
    plot_timeseries_bound!(ax, 0.0, tf, θ_max, 50; lw=1.25, abs=true)

    # Continuous-time thrust
    lbl = "\$\\arccos(\\hat e_z^{\\scriptscriptstyle"*
        "\\mathsf{T}}T_c(t)/\\|T_c(t)\\|_2)\$"
    ax.plot(t_ct, θ_ct,
            linewidth=1.5,
            color=DarkBlue,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=20,
            clip_on=false,
            label=lbl)

    # Discrete-time thrust
    lbl = "\$\\arccos(\\hat e_z^{\\scriptscriptstyle"*
        "\\mathsf{T}}T_{c,k}/\\|T_{c,k}\\|_2)\$"
    ax.plot(t_dt[1:end-1], θ_dt,
            linestyle="none",
            marker="o",
            markersize=2.5,
            markerfacecolor=Blue,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=20,
            clip_on=false,
            label=lbl)

    ax.set_ylim(0, nothing)

    leg = ax.legend(framealpha=0.8, fontsize=8,
                    loc="lower left")
    leg.set_zorder(200)

    ax.set_xlim(0, round(tf, digits=3))
    ax.set_xticks(vcat(ax.get_xticks(), round(Int, tf)))
    ax.set_xlim(0, round(tf, digits=3))

    save_figure("rocket_landing_angle.pdf", "lcvx")

    return nothing
end

"""
    plot_position(rocket, sol)

Plot the position trajectory projections.

# Arguments
- `rocket`: the rocket definition.
- `sol`: the discrete-time solution from the optimizer.
- `sim`: the continuous-time post-processed simulation.
"""
function plot_position(rocket::Rocket, sol::Solution,
                       sim::Solution)::Nothing

    # Parameters
    tf = sol.t[end]
    scale = 1/1000 # m->km
    v_scale = 3600/1000 # m/s->km/h
    thrust_scale = 0.2
    γ_gs = rocket.γ_gs
    N = length(sol.t)
    N_sim = length(sim.t)
    r_dt = sol.r*scale
    r_ct = sim.r*scale
    T_dt = sol.T
    T_ct = sim.T
    v_ct = sim.v
    v_nrm_ct = squeeze(mapslices(norm, v_ct, dims=1))*v_scale
    r_xz_dt = r_dt[[1;3], :]
    r_xz_ct = r_ct[[1;3], :]
    T_xz_dt = T_dt[[1;3], :]
    r_yz_dt = r_dt[[2;3], :]
    r_yz_ct = r_ct[[2;3], :]
    v_cmap = generate_colormap("inferno";
                               minval=minimum(v_nrm_ct),
                               maxval=maximum(v_nrm_ct))
    T_lw = 1.25
    T_lw_alpha = 0.8

    # ..:: Downrange vs. altitude plot ::..

    fig = create_figure((7.8, 4))
    gspec = fig.add_gridspec(ncols=2, nrows=1,
                             width_ratios=[0.015, 1])

    ax = setup_axis!(gspec[1, 2],
                     xlabel="Downrange [km]",
                     ylabel="Altitude [km]",
                     tight="both",
                     axis="equal")

    # Plot the continuous-time trajectory
    line_segs = Vector{RealMatrix}(undef, 0)
    line_clrs = Vector{NTuple{4, RealValue}}(undef, 0)
    overlap = 3
    for k=1:N_sim-overlap
        push!(line_segs, r_xz_ct[:, k:k+overlap]')
        push!(line_clrs, v_cmap.to_rgba(v_nrm_ct[k]))
    end
    trajectory = PyPlot.matplotlib.collections.LineCollection(
        line_segs, zorder=10,
        colors = line_clrs,
        linewidths=3,
        capstyle="round",
        joinstyle="round")
    ax.add_collection(trajectory)

    # Proxy artist for the legend
    pos_proxy = PyPlot.matplotlib.lines.Line2D(
        [0], [0],
        color=DarkBlue,
        linewidth=3,
        solid_capstyle="round")

    # Plot the discrete-time positions
    ax.plot(r_xz_dt[1, :], r_xz_dt[2, :],
            linestyle="none",
            marker="o",
            markerfacecolor=DarkBlue,
            markersize=3,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=20,
            label="\$r_k\$")

    # Highlight markers where the glideslope constraint activates
    n = [cos(γ_gs);0;-sin(γ_gs)] # Glide slope normal
    for k = 1:N
        local gs_dot = -dot(n, r_dt[:, k])
        if gs_dot<1e-12# && r_dt[3, k]>1e-10
            ax.plot(r_xz_dt[1, k], r_xz_dt[2, k];
                    marker="o",
                    markersize=3,
                    markerfacecolor=Yellow,
                    markeredgecolor="white",
                    markeredgewidth=0.2,
                    zorder=30,
                    clip_on=false)
        end
    end
    activ_proxy = PyPlot.matplotlib.lines.Line2D(
        [0], [0],
        linestyle="none",
        marker="o",
        markersize=3,
        markerfacecolor=Yellow,
        markeredgecolor="white",
        markeredgewidth=0.2)

    # Set axis limits
    y_min = minimum(r_xz_ct[2, :])-0.1
    y_max = maximum(r_xz_ct[2, :])+0.25
    x_min = minimum(r_xz_ct[1, :])-0.5
    # x_max = maximum(r_xz_ct[1, :])+0.5
    set_axis_equal(ax, (x_min, missing, y_min, y_max))

    # Plot the thrust vectors
    x_rng = collect(ax.get_xlim())
    y_rng = collect(ax.get_ylim())
    ref_sz = min(x_rng[2]-x_rng[1], y_rng[2]-y_rng[1])
    max_sz = maximum(mapslices(norm, T_xz_dt, dims=1))
    scale_factor = ref_sz/max_sz*thrust_scale
    thrust_segs = Vector{RealMatrix}(undef, 0)
    for k = 1:N-1
        thrust_whisker_base = r_xz_dt[:, k]
        thrust_whisker_tip = thrust_whisker_base+
            scale_factor*T_xz_dt[:, k]
        push!(thrust_segs, hcat(thrust_whisker_base,
                                thrust_whisker_tip)')
    end
    thrusts = PyPlot.matplotlib.collections.LineCollection(
        thrust_segs,
        zorder=5,
        colors=Green,
        linewidths=T_lw,
        alpha=T_lw_alpha,
        capstyle="round")
    ax.add_collection(thrusts)
    thrust_proxy = PyPlot.matplotlib.lines.Line2D(
        [0], [0],
        linewidth=T_lw,
        alpha=T_lw_alpha,
        color=Green,
        solid_capstyle="round")

    # Plot the glideslope
    max_x = 10 # Ridiculously large downrage
    min_y = -5 # Ridiculously far below ground
    y_vert = max_x/tan(γ_gs)
    x_verts = [0; max_x; max_x; -max_x; -max_x; 0]
    y_verts = [0; y_vert; min_y; min_y; y_vert; 0]
    ax.fill(x_verts, y_verts,
            facecolor=Red,
            alpha=0.5,
            edgecolor="none")
    ax.plot(x_verts, y_verts,
            linestyle="--",
            color=Red,
            dashes=(2, 3),
            linewidth=1,
            solid_capstyle="round",
            dash_capstyle="round")

    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    pushfirst!(handles, pos_proxy)
    pushfirst!(handles, thrust_proxy)
    push!(handles, activ_proxy)
    pushfirst!(labels, "\$r(t)\$")
    pushfirst!(labels, "\$T_{c,k}\$")
    push!(labels, "\$H_{gs}r_k=h_{gs}\$")
    leg = ax.legend(handles, labels,
                    framealpha=0.8, fontsize=8,
                    loc="upper left")
    leg.set_zorder(200)

    # Colorbar
    cbar_ax = fig.add_subplot(gspec[1, 1])
    fig.colorbar(v_cmap,
                 aspect=80,
                 label="Spped \$\\|v\\|_2\$ [km/h]",
                 orientation="vertical",
                 cax=cbar_ax)
    cbar_ax.yaxis.set_label_position("left")
    cbar_ax.yaxis.set_ticks_position("left")
    ax_pos = cbar_ax.get_position()
    ax_pos = [ax_pos.x0-0.01, ax_pos.y0, ax_pos.width, ax_pos.height]
    cbar_ax.set_position(ax_pos)

    save_figure("rocket_landing_downrange_altitude.pdf", "lcvx")

    # ..:: Crossrange vs. altitude plot ::..

    fig = create_figure((4, 4))

    ax = setup_axis!(111,
                     xlabel="Crossrange [km]",
                     ylabel="Altitude [km]",
                     tight="both",
                     axis="equal",)

    # Plot the continuous-time trajectory
    line_segs = Vector{RealMatrix}(undef, 0)
    line_clrs = Vector{NTuple{4, RealValue}}(undef, 0)
    overlap = 3
    for k=1:N_sim-overlap
        push!(line_segs, r_yz_ct[:, k:k+overlap]')
        push!(line_clrs, v_cmap.to_rgba(v_nrm_ct[k]))
    end
    trajectory = PyPlot.matplotlib.collections.LineCollection(
        line_segs, zorder=10,
        colors = line_clrs,
        linewidths=3,
        capstyle="round",
        joinstyle="round")
    ax.add_collection(trajectory)

    # Proxy artist for the legend
    pos_proxy = PyPlot.matplotlib.lines.Line2D(
        [0], [0],
        color=DarkBlue,
        linewidth=3,
        solid_capstyle="round")

    # Plot the discrete-time positions
    ax.plot(r_yz_dt[1, :], r_yz_dt[2, :],
            linestyle="none",
            marker="o",
            markerfacecolor=DarkBlue,
            markersize=3,
            markeredgecolor="white",
            markeredgewidth=0.2,
            zorder=20,
            label="\$r_k\$")

    # Highlight markers where the glideslope constraint activates
    n = [cos(γ_gs);0;-sin(γ_gs)] # Glide slope normal
    for k = 1:N
        local gs_dot = -dot(n, r_dt[:, k])
        if gs_dot<1e-12# && r_dt[3, k]>1e-10
            ax.plot(r_yz_dt[1, k], r_yz_dt[2, k];
                    marker="o",
                    markersize=3,
                    markerfacecolor=Yellow,
                    markeredgecolor="white",
                    markeredgewidth=0.2,
                    zorder=30,
                    clip_on=false)
        end
    end
    activ_proxy = PyPlot.matplotlib.lines.Line2D(
        [0], [0],
        linestyle="none",
        marker="o",
        markersize=3,
        markerfacecolor=Yellow,
        markeredgecolor="white",
        markeredgewidth=0.2)

    # Set axis limits
    y_min = minimum(r_yz_ct[2, :])-0.1
    y_max = maximum(r_xz_ct[2, :])+0.25
    _x_min = minimum(r_yz_ct[1, :])-0.5
    _x_max = maximum(r_yz_ct[1, :])+0.5
    x_min = -max(-_x_min, _x_max)
    set_axis_equal(ax, (x_min, missing, y_min, y_max))

    # Plot the thrust vectors
    x_rng = collect(ax.get_xlim())
    y_rng = collect(ax.get_ylim())
    ref_sz = min(x_rng[2]-x_rng[1], y_rng[2]-y_rng[1])
    max_sz = maximum(mapslices(norm, T_xz_dt, dims=1))
    scale_factor = ref_sz/max_sz*thrust_scale
    thrust_segs = Vector{RealMatrix}(undef, 0)
    for k = 1:N-1
        thrust_whisker_base = r_yz_dt[:, k]
        thrust_whisker_tip = thrust_whisker_base+
            scale_factor*T_xz_dt[:, k]
        push!(thrust_segs, hcat(thrust_whisker_base,
                                thrust_whisker_tip)')
    end
    thrusts = PyPlot.matplotlib.collections.LineCollection(
        thrust_segs,
        zorder=5,
        colors=Green,
        linewidths=T_lw,
        alpha=T_lw_alpha,
        capstyle="round")
    ax.add_collection(thrusts)
    thrust_proxy = PyPlot.matplotlib.lines.Line2D(
        [0], [0],
        linewidth=T_lw,
        alpha=T_lw_alpha,
        color=Green,
        solid_capstyle="round")

    # Plot the glideslope
    max_x = 10 # Ridiculously large downrage
    min_y = -5 # Ridiculously far below ground
    y_vert = max_x/tan(γ_gs)
    x_verts = [0; max_x; max_x; -max_x; -max_x; 0]
    y_verts = [0; y_vert; min_y; min_y; y_vert; 0]
    ax.fill(x_verts, y_verts,
            facecolor=Red,
            alpha=0.5,
            edgecolor="none")
    ax.plot(x_verts, y_verts,
            linestyle="--",
            color=Red,
            dashes=(2, 3),
            linewidth=1,
            solid_capstyle="round",
            dash_capstyle="round")

    # Set legend
    handles, labels = ax.get_legend_handles_labels()
    pushfirst!(handles, pos_proxy)
    pushfirst!(handles, thrust_proxy)
    push!(handles, activ_proxy)
    pushfirst!(labels, "\$r(t)\$")
    pushfirst!(labels, "\$T_{c,k}\$")
    push!(labels, "\$H_{gs}r_k=h_{gs}\$")
    leg = ax.legend(handles, labels,
                    framealpha=0.8, fontsize=8,
                    loc="upper left")
    leg.set_zorder(200)

    save_figure("rocket_landing_crossrange_altitude.pdf", "lcvx")

    return nothing
end
