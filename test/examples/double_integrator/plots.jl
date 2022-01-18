#= Double integrator trajectory plots.

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
    plot_trajectory(sol_lcvx, sol_mp)

Plot the optimal trajectory, both as obtained analytically from the maximum
principle and numerically using lossless convexification.

# Arguments
- `sol_lcvx`: numerical solution using lossless convexification.
- `sol_mp`: analytical solution using maximum principle.
- `choice`: parameter set choice, which gets appended to plot output file name.
"""
function plot_trajectory(sol_lcvx::Solution, sol_mp::Solution, choice::Int)::Nothing

    # Parameters
    t_mp = sol_mp.t
    p_mp = sol_mp.x[1, :]
    v_mp = sol_mp.x[2, :]
    u_mp = sol_mp.u[1, :]
    t_lcvx = sol_lcvx.t
    p_lcvx = sol_lcvx.x[1, :]
    v_lcvx = sol_lcvx.x[2, :]
    u_lcvx = sol_lcvx.u[1, :]
    T = t_lcvx[end]

    darken_factor = 0.2

    fig = create_figure((5, 6.5))

    axes = []

    # ..:: Position trajectory ::..
    ax = setup_axis!(311, ylabel = "Position \$x_1\$ [m]", tight = "x")
    push!(axes, ax)
    ax.tick_params(
        axis = "x",
        which = "both",
        bottom = false,
        top = false,
        labelbottom = false,
    )
    ax.plot(
        t_mp,
        p_mp,
        color = Red,
        linewidth = 2,
        solid_capstyle = "round",
        solid_joinstyle = "round",
        clip_on = false,
        zorder = 50,
        label = "Maximum principle",
    )
    ax.plot(
        t_lcvx,
        p_lcvx,
        linestyle = "none",
        marker = "o",
        markersize = 4,
        markerfacecolor = darken_color(Red, darken_factor),
        markeredgewidth = 0,
        clip_on = false,
        zorder = 50,
        label = "LCvx",
    )
    leg = ax.legend(framealpha = 0.8, fontsize = 8, loc = "upper left")
    leg.set_zorder(100)
    if choice == 1
        ax.set_ylim(0, 50)
    else
        ax.set_ylim(0, 30)
    end

    # ..:: Velocity trajectory ::..
    ax = setup_axis!(312, ylabel = "Velocity \$x_2\$ [m/s]", tight = "x")
    push!(axes, ax)
    ax.tick_params(
        axis = "x",
        which = "both",
        bottom = false,
        top = false,
        labelbottom = false,
    )
    ax.plot(
        t_mp,
        v_mp,
        color = Red,
        linewidth = 2,
        solid_capstyle = "round",
        solid_joinstyle = "round",
        clip_on = false,
        zorder = 50,
        label = "Maximum principle",
    )
    ax.plot(
        t_lcvx,
        v_lcvx,
        linestyle = "none",
        marker = "o",
        markersize = 4,
        markerfacecolor = darken_color(Red, darken_factor),
        markeredgewidth = 0,
        clip_on = false,
        zorder = 50,
        label = "LCvx",
    )
    leg = ax.legend(framealpha = 0.8, fontsize = 8, loc = "upper left")
    leg.set_zorder(100)
    if choice == 1
        ax.set_ylim(0, 10)
    else
        ax.set_ylim(-2, 6)
    end

    # ..:: Acceleration (optimal input) trajectory ::..
    ax = setup_axis!(
        313,
        xlabel = "Time [s]",
        ylabel = "Acceleration \$u\$ [m/s\$^2\$]",
        tight = "both",
    )
    push!(axes, ax)
    ax.plot(
        t_mp,
        u_mp,
        color = Blue,
        linewidth = 2,
        solid_capstyle = "round",
        solid_joinstyle = "round",
        clip_on = false,
        zorder = 50,
        label = "Maximum principle",
    )
    ax.plot(
        t_lcvx,
        u_lcvx,
        linestyle = "none",
        marker = "o",
        markersize = 4,
        markerfacecolor = darken_color(Blue, darken_factor),
        markeredgewidth = 0,
        clip_on = false,
        zorder = 50,
        label = "LCvx",
    )
    for sgn in [-1, 1]
        ax.fill(
            [0, T, T, 0, 0],
            [1, 1, 2, 2, 1] .* sgn,
            facecolor = Green,
            alpha = 0.4,
            edgecolor = "none",
            label = (sgn == -1) ? "Feasible input set" : nothing,
        )
    end

    fig.align_ylabels(axes)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[end], handles[1:end-1]...]
    labels = [labels[end], labels[1:end-1]...]
    leg = ax.legend(handles, labels, framealpha = 0.8, fontsize = 8, loc = "center left")
    leg.set_zorder(100)

    save_figure(@sprintf("double_integrator_%d.pdf", choice), "lcvx")

    return nothing
end
