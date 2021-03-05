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

using Plots
using LaTeXStrings

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

#= Constructor the quadrotor problem.

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
    tf_min = 0.1
    tf_max = 2.5
    traj = QuadrotorTrajectoryParameters(r0, rf, v0, vf, tf_min, tf_max)

    mdl = QuadrotorProblem(quad, env, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Compute the initial guess for a discrete-time trajectory.

Use straight-line interpolation and a thrust that opposes gravity ("hover").

Args:
    N: the number of temporal grid nodes.
    pbm: the trajectory problem definition.

Returns:
    x: the disrete-time state trajectory guess.
    u: the disrete-time input trajectory guess.
    p: the parameter vector guess. =#
function quadrotor_initial_guess(
    N::T_Int, pbm::TrajectoryProblem)::Tuple{T_RealMatrix,
                                             T_RealMatrix,
                                             T_RealVector}
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
end

#= Plot the trajectory evolution through SCvx iterations.

Args:
    mdl: the quadrotor problem parameters.
    history: SCP iteration data history. =#
function plot_trajectory_history(mdl::QuadrotorProblem,
                                 history::SCPHistory)::Nothing

    # Common values
    num_iter = length(history.subproblems)
    cmap = cgrad(:thermal; rev = true)
    cmap_offset = 0.1
    alph_offset = 0.3

    plot(show=false,
         aspect_ratio=:equal,
         xlabel=L"\mathrm{East~position~[m]}",
         ylabel=L"\mathrm{North~position~[m]}",
         tickfontsize=10,
         labelfontsize=10,
         size=(280, 400))

    plot_ellipsoids!(mdl.env.obs)

    # @ Draw the trajectories @
    for i = 0:num_iter

        # Extract values for the trajectory at iteration i
        if i==0
            trj = history.subproblems[1].ref
            clr = "#356397"
            alph = alph_offset
            shp = :xcross
        else
            trj = history.subproblems[i].sol
            clr = cmap[(i-1)/(num_iter-1)*(1-cmap_offset)+cmap_offset]
            alph = (i-1)/(num_iter-1)*(1-alph_offset)+alph_offset
            shp = :circle
        end
        pos = trj.xd[mdl.vehicle.id_r, :]

        plot!(pos[1, :], pos[2, :];
              reuse=true,
              legend=false,
              seriestype=:scatter,
              markershape=shp,
              markersize=6,
              markerstrokecolor="white",
              markerstrokewidth=0.3,
              color=clr,
              markeralpha=alph)
    end

    savefig("figures/scvx_quadrotor_traj_iters.pdf")

    return nothing
end

#= Plot the final converged trajectory.

Args:
    mdl: the quadrotor problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_final_trajectory(mdl::QuadrotorProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    cmap = cgrad(:thermal; rev = true)
    cmap_vel = cgrad(:thermal)
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    u_scale = 0.2

    plot(aspect_ratio=:equal,
         xlabel=L"\mathrm{East~position~[m]}",
         ylabel=L"\mathrm{North~position~[m]}",
         tickfontsize=10,
         labelfontsize=10,
         size=(280, 400),
         colorbar=:right,
         colorbar_title=L"\mathrm{Velocity~[m/s]}")

    plot_ellipsoids!(mdl.env.obs)

    # @ Draw the final continuous-time position trajectory @
    # Collect the continuous-time trajectory data
    ct_pos = T_RealMatrix(undef, 2, ct_res)
    ct_speed = T_RealVector(undef, ct_res)
    for k = 1:ct_res
        xk = sample(sol.xc, @k(ct_τ))
        @k(ct_pos) = xk[mdl.vehicle.id_r[1:2]]
        @k(ct_speed) = norm(xk[mdl.vehicle.id_v])
    end
    max_speed = maximum(ct_speed)

    # Plot the trajectory
    for k = 1:ct_res-1
        pos_beg, pos_end = @k(ct_pos), @kp1(ct_pos)
        speed_beg, speed_end = @k(ct_speed), @kp1(ct_speed)
        speed_av = 0.5*(speed_beg+speed_end)
        x = [pos_beg[1], pos_end[1]]
        y = [pos_beg[2], pos_end[2]]

        plot!(x, y;
              reuse=true,
              seriestpe=:line,
              linewidth=2,
              color=cmap_vel,
              line_z=speed_av,
              clims=(0.0, max_speed))
    end

    # @ Draw the thrust vectors @
    acc = sol.ud[mdl.vehicle.id_u, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    N = size(acc, 2)
    for k = 1:N
        base = pos[1:2, k]
        tip = base+u_scale*acc[1:2, k]
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        plot!(x, y;
              reuse=true,
              legend=false,
              seriestype=:line,
              linecolor="#db6245",
              linewidth=1.5)
    end

    # @ Draw the final discrete-time position trajectory @
    plot!(pos[1, :], pos[2, :];
          reuse=true,
          legend=false,
          seriestype=:scatter,
          markershape=:circle,
          markersize=4,
          markerstrokecolor="white",
          markerstrokewidth=0.3,
          color=cmap[1.0],
          markeralpha=1.0)


    savefig("figures/scvx_quadrotor_final_traj.pdf")

    return nothing
end

#= Plot the acceleration input norm.

Args:
    mdl: the quadrotor problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_input_norm(mdl::QuadrotorProblem,
                         sol::SCPSolution)::Nothing

    # Common
    tf = sol.p[mdl.vehicle.id_t]
    y_top = 25.0
    y_bot = 0.0
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    cmap = cgrad(:thermal; rev = true)

    plot(xlabel=L"\mathrm{Time~[s]}",
         ylabel=L"\mathrm{Acceleration~[m/s}^2\mathrm{]}",
         tickfontsize=10,
         labelfontsize=10,
         xlims=(0.0, tf),
         ylims=(y_bot, y_top),
         size=(500, 250))

    # @ Acceleration upper bound @
    bnd = mdl.vehicle.u_max
    plot_timeseries_bound!(0.0, tf, bnd, y_top-bnd)

    # @ Acceleration lower bound @
    bnd = mdl.vehicle.u_min
    plot_timeseries_bound!(0.0, tf, bnd, y_bot-bnd)

    # @ Norm of acceleration vector (continuous-time) @
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    ct_acc_vec = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_acc_nrm = T_RealVector([norm(@k(ct_acc_vec)) for k in 1:ct_res])
    plot!(ct_time, ct_acc_nrm;
          reuse=true,
          legend=false,
          seriestype=:line,
          linewidth=2,
          linecolor=cmap[1.0])

    # @ Norm of acceleration vector (discrete-time) @
    time = sol.τd*sol.p[mdl.vehicle.id_t]
    acc_vec = sol.ud[mdl.vehicle.id_u, :]
    acc_nrm = T_RealVector([norm(@k(acc_vec)) for k in 1:size(acc_vec, 2)])
    plot!(time, acc_nrm;
          reuse=true,
          legend=false,
          seriestype=:scatter,
          markershape=:circle,
          markersize=6,
          markerstrokecolor="white",
          markerstrokewidth=0.3,
          color=cmap[1.0],
          markeralpha=1.0)

    # @ Slack input (discrete-time) @
    σ = sol.ud[mdl.vehicle.id_σ, :]
    plot!(time, σ;
          reuse=true,
          legend=false,
          seriestype=:scatter,
          markershape=:hexagon,
          markersize=3,
          markerstrokecolor="white",
          markerstrokewidth=0.3,
          color="#f1d46a",
          markeralpha=1.0)

    savefig("figures/scvx_quadrotor_input.pdf")

    return nothing
end

#= Plot the acceleration input norm.

Args:
    mdl: the quadrotor problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_tilt_angle(mdl::QuadrotorProblem,
                         sol::SCPSolution)::Nothing

    # Common
    tf = sol.p[mdl.vehicle.id_t]
    y_top = 70.0
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    cmap = cgrad(:thermal; rev = true)

    plot(xlabel=L"\mathrm{Time~[s]}",
         ylabel=L"\mathrm{Tilt angle [}^\circ\mathrm{]}",
         tickfontsize=10,
         labelfontsize=10,
         xlims=(0.0, tf),
         ylims=(0.0, y_top),
         size=(500, 250))

    # @ Tilt angle upper bound @
    bnd = rad2deg(mdl.vehicle.tilt_max)
    plot_timeseries_bound!(0.0, tf, bnd, y_top-bnd)

    # @ Tilt angle (continuous-time) @
    ct_time = ct_τ*sol.p[mdl.vehicle.id_t]
    _u = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_tilt = T_RealVector([acosd(@k(_u)[3]/norm(@k(_u))) for k in 1:ct_res])
    plot!(ct_time, ct_tilt;
          reuse=true,
          legend=false,
          seriestype=:line,
          linewidth=2,
          linecolor=cmap[1.0])

    # @ Norm of acceleration vector (discrete-time) @
    time = sol.τd*sol.p[mdl.vehicle.id_t]
    _u = sol.ud[mdl.vehicle.id_u, :]
    acc_nrm = T_RealVector([acosd(@k(_u)[3]/norm(@k(_u)))
                            for k in 1:size(_u, 2)])
    plot!(time, acc_nrm;
          reuse=true,
          legend=false,
          seriestype=:scatter,
          markershape=:circle,
          markersize=6,
          markerstrokecolor="white",
          markerstrokewidth=0.3,
          color=cmap[1.0],
          markeralpha=1.0)

    savefig("figures/scvx_quadrotor_tilt.pdf")

    return nothing
end

#= Optimization algorithm convergence plot.

Args:
    mdl: the quadrotor problem parameters.
    history: SCvx iteration data history. =#
function plot_convergence(mdl::QuadrotorProblem, #nowarn
                          history::SCPHistory)::Nothing

    # Common values
    cmap = cgrad(:thermal; rev = true)

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

    plot(show=false,
         xlabel=L"\mathrm{Iteration}",
         ylabel=L"Distance from solution, $\Vert X^i-X^*\Vert_2$",
         tickfontsize=10,
         labelfontsize=10,
         yaxis=:log,
         size=(400, 300))

    plot!(iters, DX;
          reuse=true,
          legend=false,
          seriestype=:line,
          markershape=:circle,
          markersize=6,
          markerstrokecolor="white",
          markerstrokewidth=0.3,
          color=cmap[1.0],
          markeralpha=1.0,
          linewidth=2)

    savefig("figures/scvx_quadrotor_convergence.pdf")

    return nothing
end
