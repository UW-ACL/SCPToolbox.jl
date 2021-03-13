#= Quadrotor obstacle avoidance data structures and custom methods.

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

#= Quadrotor vehicle parameters. =#
struct QuadrotorParameters
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_u::T_IntRange # Indices of the thrust input vector
    id_σ::T_Int      # Index of the slack input
    id_t::T_Int      # Index of time dilation
    u_max::T_Real    # [N] Maximum thrust
    u_min::T_Real    # [N] Minimum thrust
    tilt_max::T_Real # [rad] Maximum tilt
end

#= Quadrotor flight environment. =#
struct QuadrotorEnvironmentParameters
    g::T_RealVector          # [m/s^2] Gravity vector
    obs::Vector{T_Ellipsoid} # Obstacles (ellipsoids)
    n_obs::T_Int             # Number of obstacles
end

#= Trajectory parameters. =#
struct QuadrotorTrajectoryParameters
    r0::T_RealVector # Initial position
    rf::T_RealVector # Terminal position
    v0::T_RealVector # Initial velocity
    vf::T_RealVector # Terminal velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
    γ::T_Real        # Minimum-time vs. minimum-energy tradeoff
end

#= Quadrotor trajectory optimization problem parameters all in one. =#
struct QuadrotorProblem
    vehicle::QuadrotorParameters        # The ego-vehicle
    env::QuadrotorEnvironmentParameters # The environment
    traj::QuadrotorTrajectoryParameters # The trajectory
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Constructor for the environment.

Args:
    gnrm: gravity vector norm.
    obs: array of obstacles (ellipsoids).

Returns:
    env: the environment struct. =#
function QuadrotorEnvironmentParameters(
    gnrm::T_Real,
    obs::Vector{T_Ellipsoid})::QuadrotorEnvironmentParameters

    # Derived values
    g = zeros(3)
    g[end] = -gnrm
    n_obs = length(obs)

    env = QuadrotorEnvironmentParameters(g, obs, n_obs)

    return env
end

#= Constructor for the quadrotor problem.

Returns:
    mdl: the quadrotor problem. =#
function QuadrotorProblem()::QuadrotorProblem

    # >> Quadrotor <<
    id_r = 1:3
    id_v = 4:6
    id_u = 1:3
    id_σ = 4
    id_t = 1
    u_max = 23.2
    u_min = 0.6
    tilt_max = deg2rad(60)
    quad = QuadrotorParameters(id_r, id_v, id_u, id_σ, id_t,
                               u_max, u_min, tilt_max)

    # >> Environment <<
    g = 9.81
    obs = [T_Ellipsoid(diagm([2.0; 2.0; 0.0]), [1.0; 2.0; 0.0]),
           T_Ellipsoid(diagm([1.5; 1.5; 0.0]), [2.0; 5.0; 0.0])]
    env = QuadrotorEnvironmentParameters(g, obs)

    # >> Trajectory <<
    r0 = zeros(3)
    rf = zeros(3)
    rf[1:2] = [2.5; 6.0]
    v0 = zeros(3)
    vf = zeros(3)
    tf_min = 0.0
    tf_max = 2.5
    γ = 0.0
    traj = QuadrotorTrajectoryParameters(r0, rf, v0, vf, tf_min, tf_max, γ)

    mdl = QuadrotorProblem(quad, env, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the initial discrete-time trajectory guess.

Use straight-line interpolation and a thrust that opposes gravity ("hover").

Args:
    pbm: the trajectory problem definition. =#
function quadrotor_set_initial_guess!(pbm::TrajectoryProblem)::Nothing

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
                       hover[veh.id_u] = -g
                       hover[veh.id_σ] = norm(g)
                       u = straightline_interpolate(hover, hover, N)

                       return x, u, p
                       end)

    return nothing
end

#= Plot the trajectory evolution through SCvx iterations.

Args:
    mdl: the quadrotor problem parameters.
    history: SCP iteration data history. =#
function plot_trajectory_history(mdl::QuadrotorProblem,
                                 history::SCPHistory)::Nothing

    # Common values
    num_iter = length(history.subproblems)
    algo = history.subproblems[1].algo
    cmap = get_colormap()
    cmap_offset = 0.1
    alph_offset = 0.3

    fig = create_figure((3, 4))
    ax = fig.add_subplot()

    ax.axis("equal")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    ax.set_xlabel("East position [m]")
    ax.set_ylabel("North position [m]")

    plot_ellipsoids!(ax, mdl.env.obs)

    # ..:: Draw the trajectories ::..
    for i = 0:num_iter
        # Extract values for the trajectory at iteration i
        if i==0
            trj = history.subproblems[1].ref
            alph = alph_offset
            clr = parse(RGB, "#356397")
            clr = (clr.r, clr.g, clr.b, alph)
            shp = "X"
        else
            trj = history.subproblems[i].sol
            f = (off) -> (i-1)/(num_iter-1)*(1-off)+off
            alph = f(alph_offset)
            clr = (cmap(f(cmap_offset))..., alph)
            shp = "o"
        end
        pos = trj.xd[mdl.vehicle.id_r, :]
        x, y = pos[1, :], pos[2, :]

        ax.plot(x, y,
                linestyle="none",
                marker=shp,
                markersize=5,
                markerfacecolor=clr,
                markeredgecolor=(1, 1, 1, alph),
                markeredgewidth=0.3)
    end

    save_figure("quadrotor_traj_iters", algo)

    return nothing
end

#= Plot the final converged trajectory.

Args:
    mdl: the quadrotor problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_final_trajectory(mdl::QuadrotorProblem,
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
    u_scale = 0.2

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

    plot_ellipsoids!(ax, mdl.env.obs)

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
                markeredgecolor="none")
    end

    # ..:: Draw the acceleration vector ::..
    acc = sol.ud[mdl.vehicle.id_u, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    for k = 1:N
        base = pos[1:2, k]
        tip = base+u_scale*acc[1:2, k]
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round")
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
            markeredgewidth=0.3)

    save_figure("quadrotor_final_traj", algo)

    return nothing
end

#= Plot the acceleration input norm.

Args:
    mdl: the quadrotor problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_input_norm(mdl::QuadrotorProblem,
                         sol::SCPSolution)::Nothing

    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = sol.p[mdl.vehicle.id_t]
    y_top = 25.0
    y_bot = 0.0

    fig = create_figure((5, 2.5))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [m/s\$^2\$]")

    # ..:: Acceleration bounds ::..
    bnd_max = mdl.vehicle.u_max
    bnd_min = mdl.vehicle.u_min
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # ..:: Norm of acceleration vector (continuous-time) ::..
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    ct_acc_vec = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_acc_nrm = T_RealVector([norm(@k(ct_acc_vec)) for k in 1:ct_res])
    ax.plot(ct_time, ct_acc_nrm,
            color=clr,
            linewidth=2)

    # ..:: Norm of acceleration vector (discrete-time) ::..
    time = sol.τd*sol.p[mdl.vehicle.id_t]
    acc_vec = sol.ud[mdl.vehicle.id_u, :]
    acc_nrm = T_RealVector([norm(@k(acc_vec)) for k in 1:size(acc_vec, 2)])
    ax.plot(time, acc_nrm,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr)

    # ..:: Slack input (discrete-time) ::..
    σ = sol.ud[mdl.vehicle.id_σ, :]
    ax.plot(time, σ,
            linestyle="none",
            marker="h",
            markersize=2.5,
            markeredgecolor="white",
            markeredgewidth=0.3,
            markerfacecolor="#f1d46a")

    save_figure("quadrotor_input", sol.algo)

    return nothing
end

#= Plot the acceleration input norm.

Args:
    mdl: the quadrotor problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_tilt_angle(mdl::QuadrotorProblem,
                         sol::SCPSolution)::Nothing

    # Common
    algo = sol.algo
    clr = get_colormap()(1.0)
    tf = sol.p[mdl.vehicle.id_t]
    y_top = 70.0

    fig = create_figure((5, 2.5))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)
    ax.set_ylim((0, y_top))

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tilt angle [\$^\\circ\$]")

    # ..:: Tilt angle bounds ::..
    bnd_max = rad2deg(mdl.vehicle.tilt_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)

    # ..:: Tilt angle (continuous-time) ::..
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    _u = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_tilt = T_RealVector([acosd(@k(_u)[3]/norm(@k(_u))) for k in 1:ct_res])
    ax.plot(ct_time, ct_tilt,
            color=clr,
            linewidth=2)

    # ..:: Tilt angle (discrete-time) ::..
    time = sol.τd*sol.p[mdl.vehicle.id_t]
    _u = sol.ud[mdl.vehicle.id_u, :]
    tilt = T_RealVector([acosd(@k(_u)[3]/norm(@k(_u)))
                         for k in 1:size(_u, 2)])
    ax.plot(time, tilt,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr)

    save_figure("quadrotor_tilt", sol.algo)

    return nothing
end

#= Optimization algorithm convergence plot.

Args:
    mdl: the quadrotor problem parameters.
    history: SCvx iteration data history. =#
function plot_convergence(mdl::QuadrotorProblem, #nowarn
                          history::SCPHistory)::Nothing

    # Common values
    clr = get_colormap()(1.0)

    # Compute concatenated solution vectors at each iteration
    num_iter = length(history.subproblems)
    xd = [vec(history.subproblems[i].sol.xd) for i=1:num_iter]
    ud = [vec(history.subproblems[i].sol.ud) for i=1:num_iter]
    p = [history.subproblems[i].sol.p for i=1:num_iter]
    Nnx = length(xd[1])
    Nnu = length(ud[1])
    np = length(p[1])
    X = T_RealMatrix(undef, Nnx+Nnu+np, num_iter)
    for i = 1:num_iter
        X[:, i] = vcat(xd[i], ud[i], p[i])
    end
    DX = T_RealVector([norm(X[:, i]-X[:, end]) for i=1:(num_iter-1)])
    iters = T_IntVector(1:(num_iter-1))

    fig = create_figure((4, 3))
    ax = fig.add_subplot()

    ax.set_yscale("log")
    ax.grid(linewidth=0.3, alpha=0.5, axis="y", which="major")
    ax.grid(linewidth=0.2, alpha=0.5, axis="y", which="minor", linestyle="--")
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true, axis="x")
    ax.margins(x=0.04, y=0.04)
    ax.set_xticks(1:num_iter)

    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Distance from solution, \$\\|X^i-X^*\\|_2\$")

    ax.plot(iters, DX,
            color=clr,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgewidth=0)

    save_figure("quadrotor_convergence", history.subproblems[1].algo)

    return nothing
end
