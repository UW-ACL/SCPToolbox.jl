#= This file stores the data structures and methods which define the Quadrotor
Obstacle Avoidance numerical example for SCP. =#

using Plots
using LaTeXStrings

include("../utils/types.jl")
include("../utils/helper.jl")
include("../core/scvx.jl")

# ..:: Data structures ::..

#= Quadrotor vehicle parameters. =#
struct QuadrotorParameters
    id_r::T_IntRange  # Position indices of the state vector
    id_v::T_IntRange  # Velocity indices of the state vector
    id_xt::T_Int      # Index of time dilation state
    id_u::T_IntRange  # Indices of the thrust input vector
    id_σ::T_Int       # Indices of the slack input
    id_pt::T_Int      # Index of time dilation
    u_nrm_max::T_Real # [N] Maximum thrust
    u_nrm_min::T_Real # [N] Minimum thrust
    tilt_max::T_Real  # [rad] Maximum tilt
end

#= Quadrotor flight environment. =#
struct EnvironmentParameters
    g::T_RealVector     # [m/s^2] Gravity vector
    obsN::T_Int         # Number of obstacles
    obsH::T_RealTensor  # Obstacle shapes (ellipsoids)
    obsc::T_RealMatrix  # Obstacle centers
end

#= Trajectory parameters. =#
struct TrajectoryParameters
    r0::T_RealVector # Initial position
    rf::T_RealVector # Terminal position
    v0::T_RealVector # Initial velocity
    vf::T_RealVector # Terminal velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
end

#= Quadrotor trajectory optimization problem parameters all in one. =#
struct QuadrotorProblem
    vehicle::QuadrotorParameters # The ego-vehicle
    env::EnvironmentParameters   # The environment
    traj::TrajectoryParameters   # The trajectory
end

# ..:: Constructors ::..

#= Constructor for the environment.

Args:
    gnrm: gravity vector norm.
    obsiH: array of obstacle shapes (ellipsoids).
    obsc: arrange of obstacle centers.

Returns:
    env: the environment struct. =#
function EnvironmentParameters(
    gnrm::T_Real,
    obsiH::Vector{T_RealMatrix},
    obsc::Vector{T_RealVector})::EnvironmentParameters

    obsN = length(obsiH)
    obsH = cat(obsiH...; dims=3)
    obsc = cat(obsc...; dims=2)

    # Gravity
    g = zeros(3)
    g[end] = -gnrm

    env = EnvironmentParameters(g, obsN, obsH, obsc)

    return env
end

# ..:: Public methods ::..

#= Get the i-th obstacle.

Args:
    i: the obstacle number.
    pbm: the quadrotor trajectory problem parameters.

Returns:
    H: the obtacle shape (ellipsoid).
    c: the obstacle center. =#
function get_obstacle(i::T_Int, pbm::QuadrotorProblem)::Tuple{T_RealMatrix,
                                                              T_RealVector}

    H = pbm.env.obsH[:, :, i]
    c = pbm.env.obsc[:, i]

    return H, c
end

#= Plot the trajectory evolution through SCvx iterations.

Args:
    mdl: the quadrotor problem parameters.
    history: SCvx iteration data history. =#
function plot_trajectory_history(mdl::QuadrotorProblem,
                                 history::SCvxHistory)::Nothing

    pyplot()

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

    _quadrotor__plot_obstacles!(mdl)

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
                               sol::SCvxSolution)::Nothing

    pyplot()

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

    _quadrotor__plot_obstacles!(mdl)

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
                         sol::SCvxSolution)::Nothing

    pyplot()

    # Common
    tf = sol.p[mdl.vehicle.id_pt]
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
    bnd = mdl.vehicle.u_nrm_max
    _quadrotor__plot_bound(0.0, tf, bnd, y_top-bnd)

    # @ Acceleration lower bound @
    bnd = mdl.vehicle.u_nrm_min
    _quadrotor__plot_bound(0.0, tf, bnd, y_bot-bnd)

    # @ Norm of acceleration vector (continuous-time) @
    ct_time = ct_τ*sol.p[mdl.vehicle.id_pt]
    ct_acc_vec = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_acc_nrm = T_RealVector([norm(@k(ct_acc_vec)) for k in 1:ct_res])
    plot!(ct_time, ct_acc_nrm;
          reuse=true,
          legend=false,
          seriestype=:line,
          linewidth=2,
          linecolor=cmap[1.0])

    # @ Norm of acceleration vector (discrete-time) @
    time = sol.τd*sol.p[mdl.vehicle.id_pt]
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
                         sol::SCvxSolution)::Nothing

    pyplot()

    # Common
    tf = sol.p[mdl.vehicle.id_pt]
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
    _quadrotor__plot_bound(0.0, tf, bnd, y_top-bnd)

    # @ Tilt angle (continuous-time) @
    ct_time = ct_τ*sol.p[mdl.vehicle.id_pt]
    _u = hcat([sample(sol.uc, τ)[mdl.vehicle.id_u] for τ in ct_τ]...)
    ct_tilt = T_RealVector([acosd(@k(_u)[3]/norm(@k(_u))) for k in 1:ct_res])
    plot!(ct_time, ct_tilt;
          reuse=true,
          legend=false,
          seriestype=:line,
          linewidth=2,
          linecolor=cmap[1.0])

    # @ Norm of acceleration vector (discrete-time) @
    time = sol.τd*sol.p[mdl.vehicle.id_pt]
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
function plot_convergence(mdl::QuadrotorProblem,
                          history::SCvxHistory)::Nothing

    pyplot()

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

# ..:: Private methods ::..

#= Draw the obstacles present in the environment.

Args:
    mdl: the quadrotor problem parameters. =#
function _quadrotor__plot_obstacles!(mdl::QuadrotorProblem)::Nothing
    θ = LinRange(0.0, 2*pi, 100)
    circle = hcat(cos.(θ), sin.(θ))'
    for i = 1:mdl.env.obsN
        H, c = project(get_obstacle(i, mdl)..., [1, 2])
        vertices = H\circle.+c
        obs = Shape(vertices[1, :], vertices[2, :])
        plot!(obs;
              reuse=true,
              legend=false,
              seriestype=:shape,
              color="#db6245",
              fillopacity=0.5,
              linewidth=1,
              linecolor="#26415d")
    end
end

#= Plot a bound keep-out zone.

Supposedly to show a minimum or a maximum of a quantity on a time history plot.

Args:
    x_min: the left-most value.
    x_max: the right-most value.
    y_bnd: the bound value.
    height: the "thickness" of the keep-out slab on the plot. =#
function _quadrotor__plot_bound(x_min::T_Real,
                                x_max::T_Real,
                                y_bnd::T_Real,
                                height::T_Real)::Nothing

    y_other = y_bnd+height
    x = [x_min, x_max, x_max, x_min, x_min]
    y = [y_bnd, y_bnd, y_other, y_other, y_bnd]
    infeas_region = Shape(x, y)

    plot!(infeas_region;
          reuse=true,
          legend=false,
          seriestype=:shape,
          color="#db6245",
          fillopacity=0.5,
          linewidth=0)

    plot!([x_min, x_max], [y_bnd, y_bnd];
          reuse=true,
          legend=false,
          seriestype=:line,
          color="#db6245",
          linewidth=1.75,
          linestyle=:dash)

    return nothing
end
