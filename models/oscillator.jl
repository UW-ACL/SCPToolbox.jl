#= Oscillator with dead band data structures and custom methods.

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

include("../utils/types.jl")
include("../core/problem.jl")
include("../core/scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Oscillator parameters. """
struct OscillatorParameters
    # ..:: Indices ::..
    id_r::T_Int        # Position (state)
    id_v::T_Int        # Velocity (state)
    id_aa::T_Int       # Actual acceleration (input)
    id_ar::T_Int       # Reference acceleration (input)
    id_l1r::T_IntRange # Position one-norm (parameter)
    # ..:: Mechanical parameters ::..
    ζ::T_Real          # Damping ratio
    ω0::T_Real         # [rad/s] Natural frequency
    # ..:: Control parameters ::..
    a_db::T_Real       # [m/s²] Deadband acceleration
    a_max::T_Real      # [m/s²] Maximum acceleration
end

""" Trajectory parameters. """
mutable struct OscillatorTrajectoryParameters
    r0::T_Real  # [m] Initial position
    v0::T_Real  # [m/s] Initial velocity
    tf::T_Real  # [s] Trajectory duration
    κ1::T_Real  # Sigmoid homotopy parameter
    κ2::T_Real  # Normalize homotopy parameter
end

""" Oscillator trajectory optimization problem parameters all in one. """
struct OscillatorProblem
    vehicle::OscillatorParameters        # The ego-vehicle
    traj::OscillatorTrajectoryParameters # The trajectory
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    OscillatorProblem()

Constructor for the Starship landing flip maneuver problem.

# Returns
- `mdl`: the problem definition object.
"""
function OscillatorProblem(N::T_Int)::OscillatorProblem

    # ..:: Oscillator ::..
    # >> Indices <<
    id_r = 1
    id_v = 2
    id_aa = 1
    id_ar = 2
    id_l1r = 1:N
    # >> Mechanical parameters <<
    ζ = 0.5
    ω0 = 1.0
    # >> Control parameters <<
    a_db = 0.1
    a_max = 0.2

    oscillator = OscillatorParameters(
        id_r, id_v, id_aa, id_ar, id_l1r, ζ, ω0, a_db, a_max)

    # ..:: Trajectory ::..
    r0 = 1.0
    v0 = 0.0
    tf = 10.0
    κ1 = 1e-2
    κ2 = 0.2

    traj = OscillatorTrajectoryParameters(r0, v0, tf, κ1, κ2)

    mdl = OscillatorProblem(oscillator, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    dynamics(t, k, x, u, p, pbm)

Oscillator dynamics.

Args:
- `t`: the current time (normalized).
- `k`: the current discrete-time node.
- `x`: the current state vector.
- `u`: the current input vector.
- `p`: the parameter vector.
- `pbm`: the oscillator problem description.

Returns:
- `f`: the time derivative of the state vector.
"""
function dynamics(t::T_Real, #nowarn
                  k::T_Int, #nowarn
                  x::T_RealVector,
                  u::T_RealVector,
                  p::T_RealVector, #nowarn
                  pbm::TrajectoryProblem)::T_RealVector

    # Parameters
    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj

    # Current (x, u, p) values
    r = x[veh.id_r]
    v = x[veh.id_v]
    aa = u[veh.id_aa]

    # The dynamics
    f = zeros(pbm.nx)
    f[veh.id_r] = v
    f[veh.id_v] = aa-veh.ω0^2*r-2*veh.ζ*veh.ω0*v

    # Scale for absolute time
    f *= traj.tf

    return f
end

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
    td = T_RealVector(LinRange(0.0, 1.0, pars.N))*traj.tf
    τc = T_RealVector(LinRange(0.0, 1.0, ct_res))
    tc = τc*traj.tf
    cmap = get_colormap()
    cmap_offset = 0.1
    alph_offset = 0.3
    marker_darken_factor = 0.2
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
                clr = (cmap(f(cmap_offset))..., alph)
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

    save_figure("oscillator_timeseries", algo)

    return nothing
end
