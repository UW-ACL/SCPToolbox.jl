#= Planar spacecraft rendezvous plots.

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

if isdefined(@__MODULE__, :LanguageServer)
    include("parameters.jl")
    include("../../../solvers/src/Solvers.jl")
    include("../../../solvers/src/scp.jl")
    using .Solvers
end

using Solvers
using Colors
using Printf

"""
    plot_final_trajectory(mdl, sol)

Plot the final converged trajectory.

# Arguments
- `mdl`: the planar rendezvous problem parameters.
- `sol`: the trajectory solution.
"""
function plot_final_trajectory(mdl::PlanarRendezvousProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    traj = mdl.traj
    dt_clr = rgb(generate_colormap(), 1.0)
    N = size(sol.xd, 2)
    speed = [norm(sol.xd[mdl.vehicle.id_v, k]) for k=1:N]
    v_cmap = generate_colormap("inferno"; minval=minimum(speed),
                               maxval=maximum(speed))
    u_scale = 0.2

    fig = create_figure((10, 4))

    ax = setup_axis!(; xlabel="Inertial \$x\$ [m]",
                     ylabel="Inertial \$y\$ [m]",
                     clabel="Velocity \$\\|v\\|_2\$ [m/s]",
                     cbar=v_cmap,
                     cbar_aspect=40)

    # ..:: Draw the final continuous-time position trajectory ::..

    # Collect the continuous-time trajectory data
    ct_res = 1000
    ct_τ = RealVector(LinRange(0.0, 1.0, ct_res))
    ct_pos = RealMatrix(undef, 2, ct_res)
    ct_speed = RealVector(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, ct_τ[k])
        ct_pos[:, k] = xk[mdl.vehicle.id_r]
        ct_speed[k] = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res-1
        r, v = ct_pos[:, k], ct_speed[k]
        x, y = r
        ax.plot(x, y,
                linestyle="none",
                marker="o",
                markersize=4,
                alpha=0.15,
                markerfacecolor=v_cmap.to_rgba(v),
                markeredgecolor="none",
                zorder=10)
    end

    # ..:: Discrete-time positions trajectory ::..
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
            zorder=20)

    y_min = -5
    set_axis_equal(ax, (-1.5, traj.r0[1]+1.5, y_min, missing))

    save_figure("rendezvous_planar_traj", algo)

    return nothing
end

"""
    plot_attitude(mdl, sol)

Plot the converged attitude trajectory.

# Arguments
- `mdl`: the planar rendezvous problem parameters.
- `sol`: the trajectory solution.
"""
function plot_attitude(mdl::PlanarRendezvousProblem,
                       sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    N = size(sol.xd, 2)
    tf = sol.p[veh.id_t]
    traj = mdl.traj
    ct_res = 500
    td = RealVector(LinRange(0.0, 1.0, N))*tf
    τc = RealVector(LinRange(0.0, 1.0, ct_res))
    tc = τc*tf
    clr = rgb(generate_colormap(), 1.0)

    fig = create_figure((5, 5))

    # Plot data
    data = [Dict(:ylabel=>"Angle [\$^\\circ\$]",
                 :scale=>(θ)->θ*180/pi,
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_θ),
            Dict(:ylabel=>"Angular rate [\$^\\circ\$/s]",
                 :scale=>(ω)->ω*180/pi,
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_ω)]

    axes = []

    for i = 1:length(data)
        ax = setup_axis!(length(data), 1, i;
                         xlabel="Time [s]",
                         ylabel=data[i][:ylabel],
                         tight="x")
        push!(axes, ax)

        x = data[i][:dt_y][data[i][:id], :]
        x = map(data[i][:scale], x)

        ax.plot(td, x,
                linestyle="none",
                marker="o",
                markersize=4,
                markerfacecolor=clr,
                markeredgewidth=0,
                zorder=10,
                clip_on=false)

        # >> Final continuous-time solution <<
        yc = hcat([data[i][:scale](sample(data[i][:ct_y], τ)[data[i][:id]])
                   for τ in τc]...)
        ax.plot(tc, yc[:],
                color=clr,
                linewidth=2,
                zorder=10)

    end

    fig.align_ylabels(axes)

    save_figure("rendezvous_planar_attitude", algo)

    return nothing
end

"""
    plot_attitude(mdl, sol)

Plot the converged attitude trajectory.

# Arguments
- `mdl`: the planar rendezvous problem parameters.
- `sol`: the trajectory solution.
"""
function plot_thrusts(mdl::PlanarRendezvousProblem,
                      sol::SCPSolution)::Nothing

    # Parameters
    algo = sol.algo
    veh = mdl.vehicle
    traj = mdl.traj
    N = size(sol.xd, 2)
    tf = sol.p[veh.id_t]
    ct_res = 500
    polar_resol = 1000
    td = RealVector(LinRange(0.0, 1.0, N))*tf
    τc = RealVector(LinRange(0.0, 1.0, ct_res))
    tc = τc*tf
    clr = rgb(generate_colormap(), 1.0)
    n_rcs = length(veh.id_f)
    thruster_names = [@sprintf("\$f_{%s}\$", sub) for sub in ["-", "+", "0"]]

    fig = create_figure((10, 10))

    gspec = fig.add_gridspec(ncols=2, nrows=n_rcs,
                             width_ratios=[2.2, 1])

    axes = []

    # ..:: Thrust timeseries plots ::..

    for i in veh.id_f
        ax = setup_axis!(gspec[i, 1];
                         xlabel="Time [s]",
                         ylabel=@sprintf("Thrust %s [N]", thruster_names[i]),
                         tight="x")
        push!(axes, ax)

        fi_d = sol.ud[i, :]
        fi_c = hcat([sample(sol.uc, τ)[i] for τ in τc]...)[:]

        # >> Continuous-time signal <<
        ax.plot(tc, fi_c,
                color=clr,
                linewidth=2,
                zorder=10)

        # >> Discrete-time thrust <<
        ax.plot(td, fi_d,
                linestyle="none",
                marker="o",
                markersize=4,
                markerfacecolor=clr,
                markeredgewidth=0,
                zorder=10,
                clip_on=false)

    end

    fig.align_ylabels(axes)

    # ..:: Thrust polar plots (showing deadband) ::..

    for i=1:n_rcs
        ax = setup_axis!(gspec[i, 2];
                         xlabel=@sprintf("Reference %s [N]",
                                         thruster_names[i]),
                         tight="both",
                         axis="square")

        f = sol.ud[veh.id_f[i], :]
        fr = sol.ud[veh.id_fr[i], :]

        # >> Continuous feasible (fr, f) polar <<

        fr_rng = LinRange(-veh.f_max, veh.f_max, polar_resol)
        above_db = (fr)->fr-veh.f_db
        below_db = (fr)->-veh.f_db-fr
        f_polar = map((fr)->or(above_db(fr), below_db(fr);
                               κ1=traj.κ1, κ2=traj.κ2,
                               minval=-veh.f_max-veh.f_db,
                               maxval=veh.f_max+veh.f_db)*fr, fr_rng)

        # Without deadband
        ax.plot(fr_rng, fr_rng,
                color=Red,
                linewidth=1,
                solid_capstyle="round",
                zorder=10)

        # With deadbaned
        ax.plot(fr_rng, f_polar,
                color=Green,
                linewidth=2,
                solid_capstyle="round",
                zorder=10)

        # >> The discrete-time (fr, f) trajectory values <<
        ax.plot(fr, f,
                linestyle="none",
                marker="o",
                markersize=4,
                markerfacecolor=clr,
                markeredgecolor="white",
                markeredgewidth=0.3,
                clip_on=false,
                zorder=100)

    end

    save_figure("rendezvous_planar_thrusts", algo)

    return nothing
end
