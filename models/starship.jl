#= Starship landing flip maneuver data structures and custom methods.

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

using PyPlot
using Colors

include("../utils/types.jl")
include("../core/problem.jl")
include("../core/scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Starship vehicle parameters. =#
struct StarshipParameters
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_T::T_IntRange # Indices of the thrust input vector
    id_t::T_Int      # Index of time dilation
    T_max::T_Real    # [N] Maximum thrust
    T_min::T_Real    # [N] Minimum thrust
    m::T_Real        # [kg] Vehicle mass
    tilt_max::T_Real # [rad] Maximum tilt
end

#= Starship flight environment. =#
struct StarshipEnvironmentParameters
    g::T_RealVector # [m/s^2] Gravity vector
end

#= Trajectory parameters. =#
struct StarshipTrajectoryParameters
    r0::T_RealVector # Initial position
    rf::T_RealVector # Terminal position
    v0::T_RealVector # Initial velocity
    vf::T_RealVector # Terminal velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
end

#= Starship trajectory optimization problem parameters all in one. =#
struct StarshipProblem
    vehicle::StarshipParameters        # The ego-vehicle
    env::StarshipEnvironmentParameters # The environment
    traj::StarshipTrajectoryParameters # The trajectory
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Constructor for the Starship landing flip maneuver problem.

Returns:
    mdl: the problem definition object. =#
function StarshipProblem()::StarshipProblem

    # >> Starship <<
    id_r = 1:2
    id_v = 3:4
    id_T = 1:2
    id_t = 1
    T_max = 230.0
    T_min = 6.0
    m = 2.0
    tilt_max = deg2rad(60)
    starship = StarshipParameters(id_r, id_v, id_T, id_t, T_max, T_min,
                                  m, tilt_max)

    # >> Environment <<
    g = [0.0; -9.81]
    env = StarshipEnvironmentParameters(g)

    # >> Trajectory <<
    r0 = [2.0; 6.0]
    rf = [0.0; 0.0]
    v0 = [1.0; 0.0]
    vf = zeros(2)
    tf_min = 0.0
    tf_max = 30.0
    traj = StarshipTrajectoryParameters(r0, rf, v0, vf, tf_min, tf_max)

    mdl = StarshipProblem(starship, env, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the initial discrete-time trajectory guess.

Use straight-line interpolation and a thrust that opposes gravity ("hover").

Args:
    pbm: the trajectory problem definition. =#
function starship_set_initial_guess!(pbm::TrajectoryProblem)::Nothing

    problem_set_guess!(pbm, (N, pbm) -> begin
                       veh = pbm.mdl.vehicle
                       traj = pbm.mdl.traj
                       g = pbm.mdl.env.g

                       # Parameter guess
                       p = zeros(pbm.np)
                       p[veh.id_t] = 0.5*(traj.tf_min+traj.tf_max)

                       # State guess
                       x0 = zeros(pbm.nx)
                       xf = zeros(pbm.nx)
                       x0[veh.id_r] = traj.r0
                       xf[veh.id_r] = traj.rf
                       x0[veh.id_v] = traj.v0
                       xf[veh.id_v] = traj.vf
                       x = straightline_interpolate(x0, xf, N)

                       # Input guess
                       hover = zeros(pbm.nu)
                       hover[veh.id_T] = -veh.m*g
                       u = straightline_interpolate(hover, hover, N)

                       return x, u, p
                       end)

    return nothing
end

#= Plot the final converged trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_final_trajectory(mdl::StarshipProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    dt_clr = get_colormap()(1.0)
    N = size(sol.xd, 2)
    speed = [norm(@k(sol.xd[mdl.vehicle.id_v, :])) for k=1:N]
    v_cmap = plt.get_cmap("inferno")
    v_nrm = matplotlib.colors.Normalize(vmin=minimum(speed),
                                        vmax=maximum(speed))
    v_cmap = matplotlib.cm.ScalarMappable(norm=v_nrm, cmap=v_cmap)

    fig = create_figure((3, 4))
    ax = fig.add_subplot()

    ax.axis("equal")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position [m]")
    ax.set_ylabel("North position [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity [m/s]")

    # ..:: Draw the final continuous-time position trajectory ::..
    # Collect the continuous-time trajectory data
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_pos = T_RealMatrix(undef, 2, ct_res)
    ct_speed = T_RealVector(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, @k(ct_τ))
        @k(ct_pos) = xk[mdl.vehicle.id_r[1:2]]
        @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
    end

    # Plot the trajectory
    for k = 1:ct_res-1
        r, v = @k(ct_pos), @k(ct_speed)
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

    # ..:: Draw the acceleration vector ::..
    thr = sol.ud[mdl.vehicle.id_T, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    u_hover = norm(mdl.vehicle.m*mdl.env.g)
    u_scale = 1/u_hover*(maximum(pos[2, :])-minimum(pos[2, :]))*0.1
    for k = 1:N
        base = pos[1:2, k]
        tip = base+u_scale*thr[1:2, k]
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
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
            clip_on=false,
            zorder=100)

    save_figure("starship_final_traj", algo)

    return nothing
end
