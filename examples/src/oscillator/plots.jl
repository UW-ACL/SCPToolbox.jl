#= Oscillator with deadband plots.

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

using Solvers
using Colors

"""
    plot_timeseries(mdl, history)

Plot the position and velocity timeseries.

# Arguments
- `mdl`: the oscillator problem parameters.
- `sol`: the trajectory solution.
- `history`: SCP iteration data history.
"""
function plot_timeseries(mdl::OscillatorProblem,
                         sol::SCPSolution,
                         history::SCPHistory)::Nothing

    # Common values
    num_iter = length(history.subproblems)
    pars = history.subproblems[end].def.pars
    algo = history.subproblems[end].algo
    veh = mdl.vehicle
    traj = mdl.traj
    ct_res = 500
    td = RealVector(LinRange(0.0, 1.0, pars.N))*traj.tf
    τc = RealVector(LinRange(0.0, 1.0, ct_res))
    tc = τc*traj.tf
    cmap = generate_colormap()
    cmap_offset = 0.1
    alph_offset = 0.3
    ct_clr = "#db6245"

    fig = create_figure((6, 6))

    # Plot data
    data = [Dict(:ylabel=>"Position [m]",
                 :scale=>(r)->r,
                 :dt_y=>(sol)->sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_r),
            Dict(:ylabel=>"Velocity [m/s]",
                 :scale=>(v)->v,
                 :dt_y=>(sol)->sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_v),
            Dict(:ylabel=>"Acceleration [m/s\$^2\$]",
                 :scale=>(a)->a,
                 :dt_y=>(sol)->sol.ud,
                 :ct_y=>sol.uc,
                 :id=>veh.id_aa)]

    axes = []

    for i = 1:length(data)
        ax = fig.add_subplot(length(data), 1, i)
        push!(axes, ax)

        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true, axis="x")

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(data[i][:ylabel])

        # >> Intermediate SCP subproblem solutions <<
        for j=0:num_iter
            # Extract values for the trajectory at iteration i
            if j==0
                trj = history.subproblems[1].ref
                alph = alph_offset
                clr = parse(RGB, "#356397")
                clr = rgb2pyplot(clr, a=alph)
                shp = "X"
            else
                trj = history.subproblems[j].sol
                f = (off) -> (j-1)/(num_iter-1)*(1-off)+off
                alph = f(alph_offset)
                clr = (rgb(cmap, f(cmap_offset))..., alph)
                shp = "o"
            end

            x = data[i][:dt_y](trj)[data[i][:id], :]
            x = map(data[i][:scale], x)

            ax.plot(td, x,
                    linestyle="none",
                    marker=shp,
                    markersize=4,
                    markerfacecolor=clr,
                    markeredgecolor=(1, 1, 1, alph),
                    markeredgewidth=0.3,
                    zorder=10+num_iter,
                    clip_on=false)
        end

        # >> Final continuous-time solution <<
        yc = hcat([data[i][:scale](sample(data[i][:ct_y], τ)[data[i][:id]])
                   for τ in τc]...)
        ax.plot(tc, yc[:],
                color=ct_clr,
                linewidth=3,
                alpha=0.5,
                zorder=10+num_iter-1)

    end

    fig.align_ylabels(axes)

    save_figure("oscillator_timeseries.pdf", algo)

    return nothing
end

"""
    plot_deadband(mdl, history)

Desired versus actual acceleration polar plot, which clearly visualizes the
deadband.

# Arguments
- `mdl`: the oscillator problem parameters.
- `sol`: the trajectory solution.
"""
function plot_deadband(mdl::OscillatorProblem,
                       sol::SCPSolution)::Nothing

    # Common
    algo = sol.algo
    clr = rgb(generate_colormap(), 1.0)
    veh = mdl.vehicle
    traj = mdl.traj
    resol = 1000

    aa = sol.ud[veh.id_aa, :]
    ar = sol.ud[veh.id_ar, :]

    fig = create_figure((4.5, 4))
    ax = fig.add_subplot()

    ax.axis("square")
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.autoscale(tight=true)
    ax.set_xlabel("Reference acceleration, \$a_{\\mathsf{ref}}\$ [m/s\$^2\$]")
    ax.set_ylabel("Actual acceleration, \$a_{\\mathsf{act}}\$ [m/s\$^2\$]")

    # ..:: The continuous feasible (ar, aa) polar ::..
    a_max = max(maximum(aa), maximum(ar))
    a_min = min(minimum(aa), minimum(ar))
    ar_rng = LinRange(a_min, a_max, resol)
    above_db = (ar)->ar-veh.a_db
    below_db = (ar)->-veh.a_db-ar
    aa_polar = map((ar)->or([above_db(ar); below_db(ar)],
                            κ=traj.κ1,
                            match=veh.a_max-veh.a_db,
                            normalize=veh.a_max-veh.a_db)*ar,
                   ar_rng)

    # Without deadband
    ax.plot(ar_rng, ar_rng,
            color=Red,
            linewidth=1,
            solid_capstyle="round",
            clip_on=false,
            zorder=10)

    # With deadband
    ax.plot(ar_rng, aa_polar,
            color=Green,
            linewidth=4,
            solid_capstyle="round",
            clip_on=false,
            zorder=10)

    # ..:: The discrete-time (ar, aa) trajectory values ::..
    ax.plot(ar, aa,
            linestyle="none",
            marker="o",
            markersize=4,
            markerfacecolor=clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            clip_on=false,
            zorder=100)

    save_figure("oscillator_deadband.pdf", algo)

    return nothing
end
