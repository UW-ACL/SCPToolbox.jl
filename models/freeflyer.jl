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

using PyPlot
using Colors

include("../utils/types.jl")
include("../core/scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Free-flyer vehicle parameters. """
struct FreeFlyerParameters
    id_r::T_IntRange # Position indices of the state vector
    id_v::T_IntRange # Velocity indices of the state vector
    id_q::T_IntRange # Quaternion indices of the state vector
    id_ω::T_IntRange # Angular velocity indices of the state vector
    id_T::T_IntRange # Thrust indices of the input vector
    id_M::T_IntRange # Torque indicates of the input vector
    id_t::T_Int      # Time dilation index of the parameter vector
    id_δ::T_IntRange # Room SDF indices of the parameter vector
    v_max::T_Real    # [m/s] Maximum velocity
    ω_max::T_Real    # [rad/s] Maximum angular velocity
    T_max::T_Real    # [N] Maximum thrust
    M_max::T_Real    # [N*m] Maximum torque
    m::T_Real        # [kg] Mass
    J::T_RealMatrix  # [kg*m^2] Principle moments of inertia matrix
end

""" Space station flight environment. """
struct FreeFlyerEnvironmentParameters
    obs::Vector{T_Ellipsoid}      # Obstacles (ellipsoids)
    iss::Vector{T_Hyperrectangle} # Space station flight space
    n_obs::T_Int                  # Number of obstacles
    n_iss::T_Int                  # Number of space station rooms
end

""" Trajectory parameters. """
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
    γ::T_Real        # Tradeoff weight terminal vs. running cost
    hom::T_Real      # Homotopy parameter for signed-distance function
    ε_sdf::T_Real    # Tiny weight to tighten the room SDF lower bounds
end

""" Free-flyer trajectory optimization problem parameters all in one. """
struct FreeFlyerProblem
    vehicle::FreeFlyerParameters        # The ego-vehicle
    env::FreeFlyerEnvironmentParameters # The environment
    traj::FreeFlyerTrajectoryParameters # The trajectory
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Constructors :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Constructor for the environment.

# Arguments
    iss: the space station flight corridors, defined as hyperrectangles.
    obs: array of obstacles (ellipsoids).

# Returns
    env: the environment struct.
"""
function FreeFlyerEnvironmentParameters(
    iss::Vector{T_Hyperrectangle},
    obs::Vector{T_Ellipsoid})::FreeFlyerEnvironmentParameters

    # Derived values
    n_iss = length(iss)
    n_obs = length(obs)

    env = FreeFlyerEnvironmentParameters(obs, iss, n_obs, n_iss)

    return env
end


""" Constructor for the 6-DoF free-flyer problem.

# Returns
    mdl: the free-flyer problem.
"""
function FreeFlyerProblem(N::T_Int)::FreeFlyerProblem

    # >> Environment <<
    obs_shape = diagm([1.0; 1.0; 1.0]/0.3)
    z_iss = 4.75
    obs = [T_Ellipsoid(copy(obs_shape), [8.5; -0.15; 5.0]),
           T_Ellipsoid(copy(obs_shape), [11.2; 1.84; 5.0]),
           T_Ellipsoid(copy(obs_shape), [11.3; 3.8;  4.8])]
    iss_rooms = [T_Hyperrectangle([6.0; 0.0; z_iss],
                                  1.0, 1.0, 1.5;
                                  pitch=90.0),
                 T_Hyperrectangle([7.5; 0.0; z_iss],
                                  2.0, 2.0, 4.0;
                                  pitch=90.0),
                 T_Hyperrectangle([11.5; 0.0; z_iss],
                                  1.25, 1.25, 0.5;
                                  pitch=90.0),
                 T_Hyperrectangle([10.75; -1.0; z_iss],
                                  1.5, 1.5, 1.5;
                                  yaw=-90.0, pitch=90.0),
                 T_Hyperrectangle([10.75; 1.0; z_iss],
                                  1.5, 1.5, 1.5;
                                  yaw=90.0, pitch=90.0),
                 T_Hyperrectangle([10.75; 2.5; z_iss],
                                  2.5, 2.5, 4.5;
                                  yaw=90.0, pitch=90.0)]
    env = FreeFlyerEnvironmentParameters(iss_rooms, obs)

    # >> Free-flyer <<
    id_r = 1:3
    id_v = 4:6
    id_q = 7:10
    id_ω = 11:13
    id_T = 1:3
    id_M = 4:6
    id_t = 1
    id_δ = (1:(N*env.n_iss)).+1
    v_max = 0.4
    ω_max = deg2rad(1)
    T_max = 20e-3
    M_max = 1e-4
    mass = 7.2
    J = diagm([0.1083, 0.1083, 0.1083])
    fflyer = FreeFlyerParameters(id_r, id_v, id_q, id_ω, id_T, id_M, id_t,
                                 id_δ, v_max, ω_max, T_max, M_max, mass, J)

    # >> Trajectory <<
    r0 = [6.5; -0.2; 5.0]
    v0 = [0.035; 0.035; 0.0]
    q0 = T_Quaternion(deg2rad(-40), [0.0; 1.0; 1.0])
    ω0 = zeros(3)
    rf = [11.3; 6.0; 4.5]
    vf = zeros(3)
    qf = T_Quaternion(deg2rad(0), [0.0; 0.0; 1.0])
    ωf = zeros(3)
    tf_min = 60.0
    tf_max = 200.0
    γ = 0.0
    hom = 50.0
    ε_sdf = 1e-4
    traj = FreeFlyerTrajectoryParameters(r0, rf, v0, vf, q0, qf, ω0, ωf, tf_min,
                                         tf_max, γ, hom, ε_sdf)

    mdl = FreeFlyerProblem(fflyer, env, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


""" Plot the trajectory evolution through SCP iterations.

# Arguments
    mdl: the free-flyer problem parameters.
    history: SCP iteration data history.
"""
function plot_trajectory_history(mdl::FreeFlyerProblem,
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
    ax.set_xlim((5.5, 12.5))

    ax.set_xlabel("\$x_{\\mathcal I}\$ [m]")
    ax.set_ylabel("\$y_{\\mathcal I}\$ [m]")

    plot_prisms!(ax, mdl.env.iss)
    plot_ellipsoids!(ax, mdl.env.obs)

    # ..:: Signed distance function zero-level set ::..
    z_iss = @first(history.subproblems[end].sol.xd[mdl.vehicle.id_r, :])[3]
    _freeflyer__plot_zero_levelset(ax, mdl, z_iss)

    # ..:: Draw the trajectories ::..
    for i = 0:num_iter
        # Extract values for the trajectory at iteration i
        if i==0
            trj = history.subproblems[1].ref
            alph = alph_offset
            clr = parse(RGB, "#356397")
            clr = rgb2pyplot(clr, a=alph)
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
                markeredgewidth=0.3,
                clip_on=false,
                zorder=100)
    end

    save_figure("freeflyer_traj_iters", algo)

    return nothing
end


""" Plot the final converged trajectory.

# Arguments
    mdl: the free-flyer problem parameters.
    sol: the trajectory solution.
"""
function plot_final_trajectory(mdl::FreeFlyerProblem,
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
    u_scale = 8e1

    fig = create_figure((3, 4))
    ax = fig.add_subplot()

    ax.axis("equal")
    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.set_xlim((5.5, 12.5))

    ax.set_xlabel("\$x_{\\mathcal I}\$ [m]")
    ax.set_ylabel("\$y_{\\mathcal I}\$ [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity [m/s]")

    plot_prisms!(ax, mdl.env.iss)
    plot_ellipsoids!(ax, mdl.env.obs)

    # ..:: Signed distance function zero-level set ::..
    z_iss = @first(sol.xd[mdl.vehicle.id_r, :])[3]
    _freeflyer__plot_zero_levelset(ax, mdl, z_iss)

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
                markersize=3,
                markerfacecolor=v_cmap.to_rgba(v),
                markeredgecolor="none",
                alpha=0.2,
                clip_on=false,
                zorder=100)
    end

    # ..:: Draw the thrust vector ::..
    thrust = sol.ud[mdl.vehicle.id_T, :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    for k = 1:N
        base = pos[1:2, k]
        tip = base+u_scale*thrust[1:2, k]
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#db6245",
                linewidth=1.2,
                solid_capstyle="round",
                zorder=100)
    end

    # ..:: Draw the discrete-time positions trajectory ::..
    pos = sol.xd[mdl.vehicle.id_r, :]
    x, y = pos[1, :], pos[2, :]
    ax.plot(x, y,
            linestyle="none",
            marker="o",
            markersize=2,
            markerfacecolor=dt_clr,
            markeredgecolor="white",
            markeredgewidth=0.3,
            clip_on=false,
            zorder=100)

    save_figure("freeflyer_final_traj", algo)

    return nothing
end


""" Timeseries signal plots.

# Arguments
    mdl: the free-flyer problem parameters.
    sol: the trajectory solution.
"""
function plot_timeseries(mdl::FreeFlyerProblem,
                         sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    tf = sol.p[veh.id_t]
    dt_time = sol.td*tf
    ct_time = ct_τ*tf
    clr = get_colormap()(1.0)
    xyz_clrs = ["#db6245", "#5da9a1", "#356397"]
    marker_darken_factor = 0.2
    top_scale = 1.1

    fig = create_figure((5, 5))

    # Plot data
    data = [Dict(:y_top=>top_scale*veh.T_max*1e3,
                 :bnd_max=>veh.T_max,
                 :ylabel=>"Thrust [mN]",
                 :scale=>(T)->T*1e3,
                 :dt_y=>sol.ud,
                 :ct_y=>sol.uc,
                 :id=>veh.id_T),
            Dict(:y_top=>top_scale*veh.M_max*1e3,
                 :bnd_max=>veh.M_max,
                 :ylabel=>"Torque [mN\$\\cdot\$m]",
                 :scale=>(M)->M*1e3,
                 :dt_y=>sol.ud,
                 :ct_y=>sol.uc,
                 :id=>veh.id_M),
            Dict(:y_top=>nothing,
                 :bnd_max=>nothing,
                 :ylabel=>"Attitude [\$^\\circ\$]",
                 :scale=>(q)->rad2deg.(collect(rpy(T_Quaternion(q)))),
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_q),
            Dict(:y_top=>top_scale*rad2deg(veh.ω_max),
                 :bnd_max=>veh.ω_max,
                 :ylabel=>"Angular velocity [\$^\\circ\$/s]",
                 :scale=>(ω)->rad2deg.(ω),
                 :dt_y=>sol.xd,
                 :ct_y=>sol.xc,
                 :id=>veh.id_ω)]

    for i = 1:length(data)
        ax = fig.add_subplot(2, 2, i)

        ax.grid(linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(true)
        ax.set_facecolor("white")
        ax.autoscale(tight=true)

        ax.set_xlabel("Time [s]")
        ax.set_ylabel(data[i][:ylabel])

        y_top = data[i][:y_top]
        y_max = data[i][:bnd_max]
        if !isnothing(y_max)
            y_max = data[i][:scale](y_max)
        end

        if !isnothing(y_max)
            plot_timeseries_bound!(ax, 0.0, tf, y_max, y_top-y_max)
        end

        # >> Continuous-time components <<
        yc = hcat([data[i][:scale](sample(data[i][:ct_y], τ)[data[i][:id]])
                   for τ in ct_τ]...)

        for j = 1:3
            ax.plot(ct_time, yc[j, :],
                    color=xyz_clrs[j],
                    linewidth=1)
        end

        # >> Discrete-time components <<
        yd = data[i][:dt_y][data[i][:id], :]
        yd = hcat([data[i][:scale](yd[:, k]) for k=1:size(yd,2)]...)

        for j = 1:3
            local clr = weighted_color_mean(1-marker_darken_factor,
                                            parse(RGB, xyz_clrs[j]),
                                            colorant"black")

            ax.plot(dt_time, yd[j, :],
                    linestyle="none",
                    marker="o",
                    markersize=3,
                    markeredgewidth=0.0,
                    markerfacecolor=rgb2pyplot(clr),
                    clip_on=false,
                    zorder=100)
        end

        # >> Continuous-time norm <<
        if !isnothing(y_max)
            y_nrm = T_RealVector([norm(@k(yc)) for k in 1:ct_res])
            ax.plot(ct_time, y_nrm,
                    color=clr,
                    linewidth=2,
                    linestyle=":",
                    dash_capstyle="round")
        end

        ax.set_xlim((0.0, tf))
        ax.set_ylim((minimum(yc), isnothing(y_top) ? maximum(yc) : y_top))
    end

    save_figure("freeflyer_timeseries", algo)

    return nothing
end


""" Timeseries plot of obstacle constraint values for final trajectory.

# Arguments
- `mdl`: the free-flyer problem parameters.
- `sol`: the trajectory solution.
"""
function plot_obstacle_constraints(mdl::FreeFlyerProblem,
                                   sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    env = mdl.env
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    tf = sol.p[veh.id_t]
    dt_time = sol.td*tf
    ct_time = ct_τ*tf
    cmap = get_colormap()
    xyz_clrs = ["#db6245", "#5da9a1", "#356397"]
    marker_darken_factor = 0.2

    fig = create_figure((5, 2.5))

    # ..:: Plot ISS flight space constraint ::..
    ax = fig.add_subplot(1, 2, 1)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("\$d_{\\mathrm{ISS}}(r_{\\mathcal{I}}(\\mathsf{t}))\$")

    # >> Continuous-time components <<
    yc = T_RealVector([_freeflyer__signed_distance(
        sample(sol.xc, τ)[veh.id_r], mdl) for τ in ct_τ])
    y_bot = min(-0.1, minimum(yc))
    plot_timeseries_bound!(ax, 0.0, tf, 0.0, y_bot)

    ax.plot(ct_time, yc,
            color=cmap(1.0),
            linewidth=1)

    # >> Discrete-time components <<
    yd = sol.xd[veh.id_r, :]
    yd = T_RealVector([_freeflyer__signed_distance(@k(yd), mdl)
                       for k=1:size(yd, 2)])
    ax.plot(dt_time, yd,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0.0,
            markerfacecolor=cmap(1.0),
            clip_on=false,
            zorder=100)

    ax.set_xlim((0.0, tf))
    ax.set_ylim((y_bot, maximum(yc)))

    # ..:: Plot ellipsoid obstacle constraints ::..
    ax = fig.add_subplot(1, 2, 2)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("\$\\| H_j(r_{\\mathcal{I}}(\\mathsf{t})-c_j)\\|_2\$")

    clr_offset = 0.4
    cval = (j) -> (env.n_obs==1) ? 1.0 :
        (j-1)/(env.n_obs-1)*(1-clr_offset)+clr_offset

    plot_timeseries_bound!(ax, 0.0, tf, 1.0, -1.0)

    # >> Continuous-time components <<
    for j = 1:env.n_obs
        yc = T_RealVector([env.obs[j](sample(sol.xc, τ)[veh.id_r])
                           for τ in ct_τ])

        ax.plot(ct_time, yc,
                color=cmap(cval(j)),
                linewidth=1)
    end

    # >> Discrete-time components <<
    y_top = -Inf
    for j = 1:env.n_obs
        yd = sol.xd[veh.id_r, :]
        yd = T_RealVector([env.obs[j](@k(yd)) for k=1:size(yd, 2)])
        y_top = max(y_top, maximum(yd))

        local clr = weighted_color_mean(1-marker_darken_factor,
                                        RGB(cmap(cval(j))...),
                                        colorant"black")

        ax.plot(dt_time, yd,
                linestyle="none",
                marker="o",
                markersize=3,
                markeredgewidth=0.0,
                markerfacecolor=rgb2pyplot(clr),
                clip_on=false,
                zorder=100)
    end

    ax.set_xlim((0.0, tf))
    ax.set_ylim((0.0, y_top))

    save_figure("freeflyer_obstacles", algo)

    return nothing
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


"""
    _freeflyer__signed_distance(r, mdl)

Description.

# Arguments
- `r`: the position vector.
- `mdl`: the free-flyer problem parameters.

# Returns
- `d`: the signed distance value.

"""
function _freeflyer__signed_distance(r::T_RealVector,
                                     mdl::FreeFlyerProblem)::T_Real
    room = mdl.env.iss
    d = logsumexp([1-norm((r-room[i].c)./room[i].s, Inf)
                   for i=1:mdl.env.n_iss]; t=mdl.traj.hom)
end


""" Plot the signed distance function zero-level set.

This gives a view of the space station flight space boundary that is seen by
the optimization, for a specific z-height.

# Arguments
    ax: the figure axis object.
    mdl: the free-flyer problem parameters.
    z: the z-coordinate at which to evaluate the signed distance function.
"""
function _freeflyer__plot_zero_levelset(ax::PyPlot.PyObject,
                                        mdl::FreeFlyerProblem,
                                        z::T_Real)::Nothing

    # Parameters
    env = mdl.env
    traj = mdl.traj
    room = env.iss
    xlims = (5.9, 12.1)
    ylims = (-2.6, 7.1)
    res = 100

    x = T_RealVector(LinRange(xlims..., res))
    y = T_RealVector(LinRange(ylims..., res))
    X = repeat(reshape(x, 1, :), length(y), 1)
    Y = repeat(y, 1, length(x))
    f = (x, y) -> _freeflyer__signed_distance([x; y; z], mdl)
    Z = map(f, X, Y)

    ax.contour(x, y, Z, [0.0],
               colors="#f1d46a",
               linewidths=1,
               linestyles="solid",
               zorder=10)

    return nothing
end
