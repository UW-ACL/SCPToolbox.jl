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
- `foo`: description.

# Returns
- `bar`: description.
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

    create_figure((6, 10))

    data = [Dict(:prj=>[1, 3],
                 :x_name=>"x",
                 :y_name=>"z"),
            Dict(:prj=>[1, 2],
                 :x_name=>"x",
                 :y_name=>"y"),
            Dict(:prj=>[2, 3],
                 :x_name=>"y",
                 :y_name=>"z")]

    num_plts = length(data)

    for i_plt = 1:num_plts

        # Data
        prj = data[i_plt][:prj]
        ax_x_name = data[i_plt][:x_name]
        ax_y_name = data[i_plt][:y_name]

        ax = setup_axis!(num_plts, 1, i_plt;
                         xlabel=@sprintf("LVLH \$%s\$ position [m]",
                                         ax_x_name),
                         ylabel=@sprintf("LVLH \$%s\$ position [m]",
                                         ax_y_name),
                         clabel="Velocity \$\\|v\\|_2\$ [m/s]",
                         cbar=v_cmap,
                         cbar_aspect=40)

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

    save_figure("rendezvous_trajectory_2d", algo)

    return nothing
end # function
