#= 6-Degree of Freedom free-flyer data structures and custom methods.

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
using Colors

include("../utils/types.jl")
include("../core/scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Free-flyer vehicle parameters. =#
struct FreeFlyerParameters
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_q::T_IntRange # Quaternion indices of the state vector
    id_ω::T_IntRange # Angular velocity indices of the state vector
    id_xt::T_Int     # Index of time dilation state
    id_T::T_IntRange # Thrust indices of the input vector
    id_M::T_IntRange # Torque indicates of the input vector
    id_pt::T_Int     # Index of time dilation
    v_max::T_Real    # [m/s] Maximum velocity
    ω_max::T_Real    # [rad/s] Maximum angular velocity
    T_max::T_Real    # [N] Maximum thrust
    M_max::T_Real    # [N*m] Maximum torque
    m::T_Real        # [kg] Mass
    J::T_RealMatrix  # [kg*m^2] Principle moments of inertia matrix
end

#= Space station flight environment. =#
struct FreeFlyerEnvironmentParameters
    obs::Vector{T_Ellipsoid}      # Obstacles (ellipsoids)
    iss::Vector{T_Hyperrectangle} # Space station flight space
    n_obs::T_Int                  # Number of obstacles
end

#= Trajectory parameters. =#
mutable struct FreeFlyerTrajectoryParameters
    r0::T_RealVector # Initial position
    rf::T_RealVector # Terminal position
    v0::T_RealVector # Initial velocity
    vf::T_RealVector # Terminal velocity
    q0::T_Quaternion # Initial attitude
    qf::T_Quaternion # Terminal attitude
    ω0::T_RealVector # Initial angular velocity
    ωf::T_RealVector # Terminal angular velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
    wt::T_Real       # Tradeoff weight terminal vs. running cost
    hom::T_Real      # Homotopy parameter for signed-distance function
    sdf_pwr::T_Real  # Exponent used in signed-distance function
end

#= Free-flyer trajectory optimization problem parameters all in one. =#
struct FreeFlyerProblem
    vehicle::FreeFlyerParameters        # The ego-vehicle
    env::FreeFlyerEnvironmentParameters # The environment
    traj::FreeFlyerTrajectoryParameters # The trajectory
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Constructor for the environment.

Args:
    iss: the space station flight corridors, defined as hyperrectangles.
    obs: array of obstacles (ellipsoids).

Returns:
    env: the environment struct. =#
function FreeFlyerEnvironmentParameters(
    iss::Vector{T_Hyperrectangle},
    obs::Vector{T_Ellipsoid})::FreeFlyerEnvironmentParameters

    # Derived values
    n_obs = length(obs)

    env = FreeFlyerEnvironmentParameters(obs, iss, n_obs)

    return env
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Plot the trajectory evolution through SCvx iterations.

Args:
    mdl: the free-flyer problem parameters.
    history: SCvx iteration data history. =#
function plot_trajectory_history(mdl::FreeFlyerProblem,
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

    plot_prisms!(mdl.env.iss)
    plot_ellipsoids!(mdl.env.obs)

    # @ Plot the signed distance function zero-level set @
    xlims = (6, 12)
    ylims = (-2.5, 7)
    res = 100
    z_iss = @first(history.subproblems[end].sol.xd[mdl.vehicle.id_r, :])[3]
    x = T_RealVector(LinRange(xlims..., res))
    y = T_RealVector(LinRange(ylims..., res))
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    f = (x, y) -> signed_distance(mdl.env.iss, [x; y; z_iss];
                                  t=mdl.traj.hom, a=mdl.traj.sdf_pwr)[1]
    Z = map(f, X, Y)

    contour!(x, y, Z,
             levels=[0],
             linecolor="#f1d46a",
             linewidth=1,
             colorbar=false)

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

    plot!(xlims=xlims,
          ylims=ylims)

    savefig("figures/scvx_freeflyer_traj_iters.pdf")

    return nothing
end

#= Plot the final converged trajectory.

Args:
    mdl: the free-flyer problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_final_trajectory(mdl::FreeFlyerProblem,
                               sol::SCPSolution)::Nothing

    # Common values
    cmap = cgrad(:thermal; rev = true)
    cmap_vel = cgrad(:thermal)
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    u_scale = 5e1

    plot(aspect_ratio=:equal,
         xlabel=L"$x_{\mathcal{I}}$ [m]",
         ylabel=L"$y_{\mathcal{I}}$ [m]",
         tickfontsize=10,
         labelfontsize=10,
         size=(280, 320),
         colorbar=:right,
         colorbar_title=L"$\Vert v_{\mathcal{I}}\Vert$ [m/s]")

    plot_prisms!(mdl.env.iss)
    plot_ellipsoids!(mdl.env.obs)

    # @ Plot the signed distance function zero-level set @
    xlims = (6, 12)
    ylims = (-2.5, 7)
    res = 100
    z_iss = @first(sol.xd[mdl.vehicle.id_r, :])[3]
    x = T_RealVector(LinRange(xlims..., res))
    y = T_RealVector(LinRange(ylims..., res))
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    f = (x, y) -> signed_distance(mdl.env.iss, [x; y; z_iss];
                                  t=mdl.traj.hom, a=mdl.traj.sdf_pwr)[1]
    Z = map(f, X, Y)

    contour!(x, y, Z,
             levels=[0],
             linecolor="#f1d46a",
             linewidth=1,
             colorbar=false)

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
    thrust = sol.ud[mdl.vehicle.id_T, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    N = size(thrust, 2)
    for k = 1:N
        base = pos[1:2, k]
        tip = base+u_scale*thrust[1:2, k]
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

    plot!(xlims=xlims,
          ylims=ylims)

    savefig("figures/scvx_freeflyer_final_traj.pdf")

    return nothing
end

#= Optimization algorithm convergence plot.

Args:
    mdl: the free-flyer problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_timeseries(mdl::FreeFlyerProblem,
                         sol::SCPSolution)::Nothing

    # Common values
    veh = mdl.vehicle
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    tf = sol.p[veh.id_pt]
    dt_time = sol.τd*tf
    ct_time = ct_τ*tf
    cmap = cgrad(:thermal; rev = true)
    xyz_clrs = ["#db6245", "#5da9a1", "#356397"]
    marker_darken_factor = 0.2
    top_scale = 1.1

    plot(show=false,
         tickfontsize=10,
         labelfontsize=10,
         size=(500, 500),
         layout = (2, 2))

    # Plot data
    data = [Dict(:y_top=>top_scale*veh.T_max*1e3,
                 :bnd_max=>veh.T_max,
                 :ylabel=>L"\mathrm{Thrust~[mN]}",
                 :scale=>(T)->T*1e3,
                 :dt_y=>sol.ud,
                 :ct_y=>sol.uc,
                 :id=>veh.id_T),
            Dict(:y_top=>top_scale*veh.M_max*1e3,
                 :bnd_max=>veh.M_max,
                 :ylabel=>L"Torque [mN$\cdot$m]",
                 :scale=>(M)->M*1e3,
                 :dt_y=>sol.ud,
                 :ct_y=>sol.uc,
                 :id=>veh.id_M),
            Dict(:y_top=>nothing,
                 :bnd_max=>nothing,
                 :ylabel=>L"Attitude [$^\circ$]",
                 :scale=>(q)->rad2deg.(collect(rpy(T_Quaternion(q)))),
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_q),
            Dict(:y_top=>top_scale*rad2deg(veh.ω_max),
                 :bnd_max=>veh.ω_max,
                 :ylabel=>L"Angular velocity [$^\circ$/s]",
                 :scale=>(ω)->rad2deg.(ω),
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_ω)]

    for i = 1:length(data)
        y_top = data[i][:y_top]
        y_max = data[i][:bnd_max]
        if !isnothing(y_max)
            y_max = data[i][:scale](y_max)
        end
        plot!(subplot=i,
              xlabel=L"\mathrm{Time~[s]}",
              ylabel=data[i][:ylabel])
        if !isnothing(y_max)
            plot_timeseries_bound!(0.0, tf, y_max, y_top-y_max; subplot=i)
        end
        # @ Continuous-time components @
        yc = hcat([data[i][:scale](sample(data[i][:ct_y], τ)[data[i][:id]])
                   for τ in ct_τ]...)
        for j = 1:3
            plot!(ct_time, yc[j, :];
                  subplot=i,
                  reuse=true,
                  legend=false,
                  seriestype=:line,
                  linewidth=1,
                  color=xyz_clrs[j])
        end
        # @ Discrete-time components @
        yd = data[i][:dt_y][data[i][:id], :]
        yd = hcat([data[i][:scale](yd[:, k]) for k=1:size(yd,2)]...)
        for j = 1:3
            clr = weighted_color_mean(1-marker_darken_factor,
                                      parse(RGB, xyz_clrs[j]),
                                      colorant"black")
            plot!(dt_time, yd[j, :];
                  subplot=i,
                  reuse=true,
                  legend=false,
                  seriestype=:scatter,
                  markershape=:circle,
                  markersize=4,
                  markerstrokewidth=0.0,
                  color=clr,
                  markeralpha=1.0)
        end
        # @ Continuous-time norm @
        if !isnothing(y_max)
            y_nrm = T_RealVector([norm(@k(yc)) for k in 1:ct_res])
            plot!(ct_time, y_nrm;
                  subplot=i,
                  reuse=true,
                  legend=false,
                  seriestype=:line,
                  linewidth=2,
                  linestyle=:dash,
                  color="black")
        end
        plot!(subplot=i,
              xlims=(0.0, tf),
              ylims=(minimum(yc),
                     isnothing(y_top) ? maximum(yc) : y_top))
    end

    savefig("figures/scvx_freeflyer_timeseries.pdf")

    return nothing
end

#= Optimization algorithm convergence plot.

Args:
    mdl: the free-flyer problem parameters.
    history: SCvx iteration data history. =#
function plot_convergence(mdl::FreeFlyerProblem, #nowarn
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
    DX = max.(T_RealVector([norm(X[:, i]-X[:, end]) for i=1:(num_iter-1)]), eps())
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

    savefig("figures/scvx_freeflyer_convergence.pdf")

    return nothing
end

#= Timeseries plot of obstacle constraint values for final trajectory.

Args:
    mdl: the free-flyer problem parameters.
    sol: the trajectory solution output by SCvx. =#
function plot_obstacle_constraints(mdl::FreeFlyerProblem,
                                   sol::SCPSolution)::Nothing

    # Common values
    veh = mdl.vehicle
    env = mdl.env
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    tf = sol.p[veh.id_pt]
    dt_time = sol.τd*tf
    ct_time = ct_τ*tf
    cmap = cgrad(:thermal; rev = true)
    xyz_clrs = ["#db6245", "#5da9a1", "#356397"]
    marker_darken_factor = 0.2

    plot(show=false,
         tickfontsize=10,
         labelfontsize=10,
         size=(500, 250),
         layout = (1, 2))

    # ..:: Plot ISS flight space constraint ::..
    y_max = 0.0
    plot!(subplot=1,
          xlabel=L"\mathrm{Time~[s]}",
          ylabel=L"d_{\mathrm{ISS}}(r_{\mathcal{I}}(t))")
    # >> Continuous-time components <<
    yc = T_RealVector([signed_distance(env.iss,
                                       sample(sol.xc, τ)[veh.id_r];
                                       t=mdl.traj.hom, a=mdl.traj.sdf_pwr)[1]
                       for τ in ct_τ])
    y_top = max(0.1, maximum(yc))
    plot_timeseries_bound!(0.0, tf, y_max, y_top-y_max; subplot=1)
    plot!(ct_time, yc;
          subplot=1,
          reuse=true,
          legend=false,
          seriestype=:line,
          linewidth=1,
          color=cmap[1.0])
    # >> Discrete-time components <<
    yd = sol.xd[veh.id_r, :]
    yd = T_RealVector([signed_distance(env.iss, @k(yd); t=mdl.traj.hom,
                                       a=mdl.traj.sdf_pwr)[1]
                       for k=1:size(yd, 2)])
    plot!(dt_time, yd;
          subplot=1,
          reuse=true,
          legend=false,
          seriestype=:scatter,
          markershape=:circle,
          markersize=4,
          markerstrokewidth=0.0,
          color=cmap[1.0],
          markeralpha=1.0)
    plot!(subplot=1,
          xlims=(0.0, tf),
          ylims=(minimum(yc), y_top))

    # ..:: Plot ellipsoid obstacle constraints ::..
    clr_offset = 0.4
    clr_map = (j) -> (env.n_obs==1) ? 1.0 :
        (j-1)/(env.n_obs-1)*(1-clr_offset)+clr_offset
    y_min = 1.0
    y_bot = 0.0
    plot!(subplot=2,
          xlabel=L"\mathrm{Time~[s]}",
          ylabel=L"\Vert H_j(r_{\mathcal{I}}(t)-c_j)\Vert_2")
    plot_timeseries_bound!(0.0, tf, y_min, y_bot-y_min; subplot=2)
    # >> Continuous-time components <<
    for j = 1:env.n_obs
        yc = T_RealVector([env.obs[j](sample(sol.xc, τ)[veh.id_r])
                           for τ in ct_τ])
        plot!(ct_time, yc;
              subplot=2,
              reuse=true,
              legend=false,
              seriestype=:line,
              linewidth=1,
              color=cmap[clr_map(j)])
    end
    # >> Discrete-time components <<
    y_top = -Inf
    for j = 1:env.n_obs
        yd = sol.xd[veh.id_r, :]
        yd = T_RealVector([env.obs[j](@k(yd)) for k=1:size(yd, 2)])
        y_top = max(y_top, maximum(yd))
        plot!(dt_time, yd;
              subplot=2,
              reuse=true,
              legend=false,
              seriestype=:scatter,
              markershape=:circle,
              markersize=4,
              markerstrokewidth=0.0,
              color=cmap[clr_map(j)],
              markeralpha=1.0)
    end
    plot!(subplot=2,
          xlims=(0.0, tf),
          ylims=(y_bot, y_top))

    savefig("figures/scvx_freeflyer_obstacles.pdf")

    return nothing
end
