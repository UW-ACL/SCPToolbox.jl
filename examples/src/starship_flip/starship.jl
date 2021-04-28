#= Starship landing flip maneuver data structures and custom methods.

Disclaimer: the data in this example is obtained entirely from publicly
available information, e.g. on reddit.com/r/spacex, nasaspaceflight.com, and
spaceflight101.com. No SpaceX engineers were involved in the creation of this
code.

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

include("../core/scp.jl")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Data structures ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#= Starship vehicle parameters. =#
struct StarshipParameters
    # ..:: Indices ::..
    id_r::T_IntRange   # Position indices of the state vector
    id_v::T_IntRange   # Velocity indices of the state vector
    id_θ::T_Int        # Tilt angle index of the state vector
    id_ω::T_Int        # Tilt rate index of the state vector
    id_m::T_Int        # Mass index of the state vector
    id_δd::T_Int       # Delayed gimbal angle index of the state vector
    id_T::T_Int        # Thrust index of the input vector
    id_δ::T_Int        # Gimbal angle index of the input vector
    id_δdot::T_Int     # Gimbal rate index of the input vector
    id_t1::T_Int       # First phase duration index of parameter vector
    id_t2::T_Int       # Second phase duration index of parameter vector
    id_xs::T_IntRange  # State at phase switch indices of parameter vector
    # ..:: Body axes ::..
    ei::T_Function     # Lateral body axis in body or world frame
    ej::T_Function     # Longitudinal body axis in body or world frame
    # ..:: Mechanical parameters ::..
    lcg::T_Real        # [m] CG location (from base)
    lcp::T_Real        # [m] CP location (from base)
    m::T_Real          # [kg] Total mass
    J::T_Real          # [kg*m^2] Moment of inertia about CG
    # ..:: Aerodynamic parameters ::..
    CD::T_Real         # [kg/m] Overall drag coefficient 0.5*ρ*cd*A
    # ..:: Propulsion parameters ::..
    T_min1::T_Real     # [N] Minimum thrust (one engine)
    T_max1::T_Real     # [N] Maximum thrust (one engine)
    T_min3::T_Real     # [N] Minimum thrust (three engines)
    T_max3::T_Real     # [N] Maximum thrust (three engines)
    αe::T_Real         # [s/m] Mass depletion propotionality constant
    δ_max::T_Real      # [rad] Maximum gimbal angle
    δdot_max::T_Real   # [rad/s] Maximum gimbal rate
    rate_delay::T_Real # [s] Delay for approximate rate constraint
end

#= Starship flight environment. =#
struct StarshipEnvironmentParameters
    ex::T_RealVector # Horizontal "along" axis
    ey::T_RealVector # Vertical "up" axis
    g::T_RealVector  # [m/s^2] Gravity vector
end

#= Trajectory parameters. =#
mutable struct StarshipTrajectoryParameters
    r0::T_RealVector # [m] Initial position
    v0::T_RealVector # [m/s] Initial velocity
    θ0::T_Real       # [rad] Initial tilt angle
    vs::T_RealVector # [m/s] Phase switch velocity
    θs::T_Real       # [rad] Phase switch tilt angle
    vf::T_RealVector # [m/s] Terminal velocity
    tf_min::T_Real   # Minimum flight time
    tf_max::T_Real   # Maximum flight time
    γ_gs::T_Real     # [rad] Maximum glideslope (measured from vertical)
    θmax2::T_Real    # [rad] Maximum tilt for terminal descent phase
    τs::T_Real       # Normalized time end of first phase
    hs::T_Real       # [m] Phase switch altitude guess
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

    # ..:: Environment ::..
    ex = [1.0; 0.0]
    ey = [0.0; 1.0]
    g0 = 9.81 # [m/s^2] Gravitational acceleration
    g = -g0*ey
    env = StarshipEnvironmentParameters(ex, ey, g)

    # ..:: Starship ::..
    # >> Indices <<
    id_r = 1:2
    id_v = 3:4
    id_θ = 5
    id_ω = 6
    id_m = 7
    id_δd = 8
    id_T = 1
    id_δ = 2
    id_δdot = 3
    id_t1 = 1
    id_t2 = 2
    id_xs = (1:id_δd).+2
    # >> Body axes <<
    ei = (θ) -> cos(θ)*[1.0; 0.0]+sin(θ)*[0.0; 1.0]
    ej = (θ) -> -sin(θ)*[1.0; 0.0]+cos(θ)*[0.0; 1.0]
    # >> Mechanical parameters <<
    rs = 4.5 # [m] Fuselage radius
    ls = 50.0 # [m] Fuselage height
    m = 120e3
    lcg = 0.4*ls
    lcp = 0.45*ls
    J = 1/12*m*(6*rs^2+ls^2)
    # >> Aerodynamic parameters <<
    vterm = 85 # [m/s] Terminal velocity (during freefall)
    CD = m*g0/vterm^2
    CD *= 1.2 # Fudge factor
    # >> Propulsion parameters <<
    Isp = 330 # [s] Specific impulse
    T_min1 = 880e3 # [N] One engine min thrust
    T_max1 = 2210e3 # [N] One engine max thrust
    T_min3 = 3*T_min1
    T_max3 = 3*T_max1
    αe = -1/(Isp*g0)
    δ_max = deg2rad(10.0)
    δdot_max = 2*δ_max
    rate_delay = 0.05

    starship = StarshipParameters(
        id_r, id_v, id_θ, id_ω, id_m, id_δd, id_T, id_δ, id_δdot, id_t1,
        id_t2, id_xs, ei, ej, lcg, lcp, m, J, CD, T_min1, T_max1, T_min3,
        T_max3, αe, δ_max, δdot_max, rate_delay)

    # ..:: Trajectory ::..
    # Initial values
    r0 = 100.0*ex+600.0*ey
    v0 = -vterm*ey
    θ0 = deg2rad(90.0)
    # Phase switch (guess) values
    θs = deg2rad(-10.0)
    vs = -10.0*ey
    # Terminal values
    vf = -0.1*ey
    tf_min = 0.0
    tf_max = 40.0
    γ_gs = deg2rad(27.0)
    θmax2 = deg2rad(15.0)
    τs = 0.5
    hs = 100.0
    traj = StarshipTrajectoryParameters(r0, v0, θ0, vs, θs, vf, tf_min,
                                        tf_max, γ_gs, θmax2, τs, hs)

    mdl = StarshipProblem(starship, env, traj)

    return mdl
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Public methods :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

"""
    dynamics(t, k, x, u, p, pbm[; no_aero_torques])

Starship vehicle dynamics.

Args:
- `t`: the current time (normalized).
- `k`: the current discrete-time node.
- `x`: the current state vector.
- `u`: the current input vector.
- `p`: the parameter vector.
- `pbm`: the Starship landing flip problem description.
- `no_aero_torques`: (optional) whether to omit torques generated by lift and
  drag.

Returns:
- `f`: the time derivative of the state vector.
"""
function dynamics(t::T_Real,
                  k::T_Int,
                  x::T_RealVector,
                  u::T_RealVector,
                  p::T_RealVector,
                  pbm::TrajectoryProblem;
                  no_aero_torques::T_Bool=false)::T_RealVector

    # Parameters
    veh = pbm.mdl.vehicle
    env = pbm.mdl.env
    traj = pbm.mdl.traj

    # Current (x, u, p) values
    v = x[veh.id_v]
    θ = x[veh.id_θ]
    ω = x[veh.id_ω]
    m = x[veh.id_m]
    δd = x[veh.id_δd]
    T = u[veh.id_T]
    δ = u[veh.id_δ]
    tdil = ((t<=traj.τs) ? p[veh.id_t1]/traj.τs :
            p[veh.id_t2]/(1-traj.τs))

    # Derived quantities
    ℓeng = -veh.lcg
    ℓcp = veh.lcp-veh.lcg
    ei = veh.ei(θ)
    ej = veh.ej(θ)
    Tv = T*(-sin(δ)*ei+cos(δ)*ej)
    MT = ℓeng*T*sin(δ)
    D = -veh.CD*norm(v)*v
    if !no_aero_torques
        MD = -ℓcp*dot(D, ei)
    else
        MD = 0.0
    end

    # The dynamics
    f = zeros(pbm.nx)
    f[veh.id_r] = v
    f[veh.id_v] = (Tv+D)/veh.m+env.g
    f[veh.id_θ] = ω
    f[veh.id_ω] = (MT+MD)/veh.J
    f[veh.id_m] = veh.αe*T
    f[veh.id_δd] = (δ-δd)/veh.rate_delay

    # Scale for time
    f *= tdil

    return f
end


""" Compute the initial trajectory guess.

This uses a simple bang-bang control strategy for the flip maneuver. Once
Starship is upright, convex optimization is used to find the terminal descent
trajectory by approximatin Starship as a double-integrator (no attitude, no
aerodynamics).

Args:
* `N`: the number of discrete-time grid nodes.
* `pbm`: the trajectory problem structure.

Returns:
* `x_guess`: the state trajectory initial guess.
* `u_guess`: the input trajectory initial guess.
* `p_guess`: the parameter vector initial guess.
"""
function starship_initial_guess(
    N::T_Int,
    pbm::TrajectoryProblem)::Tuple{T_RealMatrix,
                                   T_RealMatrix,
                                   T_RealVector}

    @printf("Computing initial guess .")

    # Parameters
    veh = pbm.mdl.vehicle
    traj = pbm.mdl.traj
    env = pbm.mdl.env

    # Normalized time grid
    τ_grid = T_RealVector(LinRange(0.0, 1.0, N))
    id_phase1 = findall(τ_grid.<=traj.τs)
    id_phase2 = T_IntVector(id_phase1[end]:N)

    # Initialize empty trajectory guess
    x_guess = zeros(pbm.nx, N)
    u_guess = zeros(pbm.nu, N)

    ######################################################
    # Phase 1: flip ######################################
    ######################################################

    # Simple guess control strategy
    # Gimbal bang-bang drive θ0 to θs at min 3-engine thrust
    flip_ac = veh.lcg/veh.J*veh.T_min3*sin(veh.δ_max)
    flip_ts = sqrt((traj.θ0-traj.θs)/flip_ac)
    flip_ctrl = (t, pbm) -> begin
        veh = pbm.mdl.vehicle
        T = veh.T_min3
        ts = flip_ts
        if t<=ts
            δ = veh.δ_max
        elseif t>ts && t<=2*ts
            δ = -veh.δ_max
        else
            δ = 0.0
        end
        u = zeros(pbm.nu)
        u[veh.id_T] = T
        u[veh.id_δ] = δ
        return u
    end

    # Dynamics with guess control
    flip_f = (t, x, pbm) -> begin
        veh = pbm.mdl.vehicle
        traj = pbm.mdl.traj
        u = flip_ctrl(t, pbm)
        k = max(floor(T_Int, t/(N-1))+1, N)
        p = zeros(pbm.np)
        p[veh.id_t1] = traj.τs
        p[veh.id_t2] = 1-traj.τs
        dxdt = dynamics(t, k, x, u, p, pbm; no_aero_torques=true)
        return dxdt
    end

    # Initial condition
    x10 = zeros(pbm.nx)
    x10[veh.id_r] = traj.r0
    x10[veh.id_v] = traj.v0
    x10[veh.id_θ] = traj.θ0
    x10[veh.id_δd] = veh.δ_max

    # Propagate the flip dynamics under the guess control
    t_θcst = 10.0
    tf = 2*flip_ts+t_θcst
    t = T_RealVector(LinRange(0.0, tf, 5000))
    x1 = rk4((t, x) -> flip_f(t, x, pbm), x10, t; full=true)

    # Find crossing of terminal vertical velocity
    vs = dot(traj.vs, env.ey)
    k_0x = findfirst(x1[veh.id_v, :]'*env.ey.>=vs)
    if isnothing(k_0x)
        msg = string("no terminal velocity crossing, ",
                     "increase time of flight (t_θcst).")
        error = ArgumentError(msg)
        throw(error)
    end
    t = @k(t, 1, k_0x)
    t1 = t[end]
    x1 = @k(x1, 1, k_0x)

    # Populate trajectory guess first phase
    τ2t = (τ) -> τ/traj.τs*t1
    x1c = T_ContinuousTimeTrajectory(t, x1, :linear)
    @k(x_guess, id_phase1) = hcat([
        sample(x1c, τ2t(τ)) for τ in τ_grid[id_phase1]]...)
    @k(u_guess, id_phase1) = hcat([
        flip_ctrl(τ2t(τ), pbm) for τ in τ_grid[id_phase1]]...)

    @printf(".")

    ######################################################
    # Phase 2: terminal descent ##########################
    ######################################################

    # Get the transition state
    xs = sample(x1c, τ2t(τ_grid[id_phase1[end]]))
    traj.hs = dot(xs[veh.id_r], env.ey)

    # Discrete time grid
    τ2 = τ_grid[id_phase2].-τ_grid[id_phase2[1]]
    N2 = length(τ2)
    tdil = (t2) -> t2/(1-traj.τs) # Time dilation amount

    # State and control dims for simple system
    nx = 4
    nu = 2

    # LTI state space matrices
    A_lti = [zeros(2,2) I(2); zeros(2, 4)]
    B_lti = [zeros(2,2); I(2)/veh.m]
    r_lti = [zeros(2); env.g]

    # Matrix indices in concatenated vector
    idcs_A = (1:nx*nx)
    idcs_Bm = (1:nx*nu).+idcs_A[end]
    idcs_Bp = (1:nx*nu).+idcs_Bm[end]
    idcs_r = (1:nx).+idcs_Bp[end]

    # Concatenated time derivative for propagation
    derivs = (t, V, Δt, tdil) -> begin
        # Get current values
        Phi = reshape(V[idcs_A], (nx, nx))
        σm = (Δt-t)/Δt
        σp = t/Δt

        # Apply time dilation to integrate in absolute time
        _A = tdil*A_lti
        _B = tdil*B_lti
        _r = tdil*r_lti

        # Compute derivatives
        iPhi = Phi\I(nx)
        dPhidt = _A*Phi
        dBmdt = iPhi*_B*σm
        dBpdt = iPhi*_B*σp
        drdt = iPhi*_r

        dVdt = [vec(dPhidt); vec(dBmdt); vec(dBpdt); drdt]

        return dVdt
    end

    # Continuous to discrete time dynamics conversion function
    discretize = (t2) -> begin
        # Propagate the dynamics over a single time interval
        Δt = τ2[2]-τ2[1]
        F = (t, V) -> derivs(t, V, Δt, tdil(t2))
        t_grid = T_RealVector(LinRange(0, Δt, 100))
        V0 = zeros(idcs_r[end])
        V0[idcs_A] = vec(I(nx))
        V = rk4(F, V0, t_grid)

        # Get the raw RK4 results
        AV = V[idcs_A]
        BmV = V[idcs_Bm]
        BpV = V[idcs_Bp]
        rV = V[idcs_r]

        # Extract the discrete-time update matrices for this time interval
        A = reshape(AV, (nx, nx))
        Bm = A*reshape(BmV, (nx, nu))
        Bp = A*reshape(BpV, (nx, nu))
        r = A*rV

        return A, Bm, Bp, r
    end

    # Variable scaling
    zero_intvl_tol = sqrt(eps())
    Tmax_x = veh.T_max1*sin(traj.θmax2)

    update_scale! = (S, c, i, min, max) -> begin
        if min > max
            min, max = max, min
        end
        if (max-min)>zero_intvl_tol
            S[i, i] = max-min
            c[i] = min
        end
    end

    Sx, cx = T_RealMatrix(I(nx)), zeros(nx)
    Su, cu = T_RealMatrix(I(nu)), zeros(nu)

    update_scale!(Sx, cx, 1, 0, xs[veh.id_r[1]])
    update_scale!(Sx, cx, 2, 0, xs[veh.id_r[2]])
    update_scale!(Sx, cx, 3, 0, xs[veh.id_v[1]])
    update_scale!(Sx, cx, 4, 0, xs[veh.id_v[2]])
    update_scale!(Su, cu, 1, -Tmax_x, Tmax_x)
    update_scale!(Su, cu, 2, veh.T_min1, veh.T_max1)

    # Solver for a trajectory, given a time of flight
    solve_trajectory = (t2) -> begin
        # >> Formulate the convex optimization problem <<
        cvx = Model()
        set_optimizer(cvx, ECOS.Optimizer)
        set_optimizer_attribute(cvx, "verbose", 0)

        # Decision variables
        xh = @variable(cvx, [1:nx, 1:N2], base_name="xh")
        uh = @variable(cvx, [1:nu, 1:N2], base_name="uh")
        x = Sx*xh.+cx
        u = Su*uh.+cu

        # Boundary conditions
        x0 = zeros(nx)
        xf = zeros(nx)
        x0[1:2] = xs[veh.id_r]
        x0[3:4] = xs[veh.id_v]
        xf[3:4] = traj.vf
        @constraint(cvx, @first(x) .== x0)
        @constraint(cvx, @last(x) .== xf)

        # Dynamics
        A, Bm, Bp, r = discretize(t2)
        for k = 1:N2-1
            xk, xkp1, uk, ukp1 = @k(x), @kp1(x), @k(u), @kp1(u)
            @constraint(cvx, xkp1 .== A*xk+Bm*uk+Bp*ukp1+r)
        end

        # Input constraints
        C = T_ConvexConeConstraint
        acc! = add_conic_constraint!
        for k = 1:N2
            uk = @k(u)
            acc!(cvx, C(vcat(veh.T_max1, uk), :soc))
            acc!(cvx, C(veh.T_min1-dot(uk, env.ey), :nonpos))
            acc!(cvx, C(vcat(dot(uk, env.ey)/cos(traj.θmax2), uk), :soc))
        end

        # State constraints
        for k = 1:N2
            xk = @k(x)
            rk = xk[1:2]
            # acc!(cvx, C(vcat(dot(rk, env.ey)/cos(traj.γ_gs), rk), :soc))
            @constraint(cvx, dot(rk, env.ey)>=0)
        end

        # Cost function
        set_objective_function(cvx, 0.0)
        set_objective_sense(cvx, MOI.MIN_SENSE)

        # >> Solve <<
        optimize!(cvx)

        # Return the solution
        x = value.(x)
        u = value.(u)
        status = termination_status(cvx)

        return x, u, status
    end

    # Find the first (smallest) time that gives a feasible trajectory
    t2_range = [10.0, 40.0]
    Δt2 = 1.0 # Amount to increment t2 guess by
    t2, x2, T2 = t2_range[1], nothing, nothing
    while true
        @printf(".")
        _x, _u, status = solve_trajectory(t2)
        if status==MOI.OPTIMAL || status==MOI.ALMOST_OPTIMAL
            x2 = _x
            T2 = _u
            break
        end
        t2 += Δt2
        if t2>t2_range[2]
            msg = string("could not find a terminal ",
                         "descent time of flight.")
            err = SCPError(0, SCP_BAD_PROBLEM, msg)
            throw(err)
        end
    end

    # Add terminal descent to initial guess
    x_guess[veh.id_r, id_phase2] = x2[1:2, :]
    x_guess[veh.id_v, id_phase2] = x2[3:4, :]
    _tdil = tdil(t2)
    m20 = x_guess[veh.id_m, id_phase2[1]]
    for k = 1:N2
        Tk = @k(T2)
        j = @k(id_phase2)
        x_guess[veh.id_θ, j] = -atan(Tk[1], Tk[2])
        u_guess[veh.id_T, j] = norm(Tk)
        if k>1
            # Angular velocity
            Δθ = x_guess[veh.id_θ, j]-x_guess[veh.id_θ, j-1]
            Δt = (τ2[k]-τ2[k-1])*_tdil
            x_guess[veh.id_ω, j-1] = Δθ/Δt
            # Mass
            x_guess[veh.id_m, j] = m20+trapz(
                veh.αe*u_guess[veh.id_T, id_phase2[1:k]],
                τ2[1:k]*_tdil)
        end
    end

    # Parameter guess
    p_guess = T_RealVector(undef, pbm.np)
    p_guess[veh.id_t1] = t1
    p_guess[veh.id_t2] = t2
    p_guess[veh.id_xs] = xs

    @printf(". done\n")

    return x_guess, u_guess, p_guess
end

""" Plot the trajectory evolution through SCP iterations.

Args:
    mdl: the quadrotor problem parameters.
    history: SCP iteration data history.
"""
function plot_trajectory_history(mdl::StarshipProblem,
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

    ax.set_xlabel("Downrange [m]")
    ax.set_ylabel("Altitude [m]")

    # Draw the glide slope constraint
    _starship__plot_glideslope(ax, mdl)

    # ..:: Draw the trajectories ::..
    for i=0:num_iter
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

    save_figure("starship_traj_iters", algo)

    return nothing
end

#= Plot the final converged trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution output by SCP. =#
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

    ax.set_xlabel("Downrange [m]")
    ax.set_ylabel("Altitude [m]")

    # Colorbar for velocity norm
    plt.colorbar(v_cmap,
                 aspect=40,
                 label="Velocity [m/s]")

    # Draw the glide slope constraint
    _starship__plot_glideslope(ax, mdl)

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
    T = sol.ud[mdl.vehicle.id_T, :]
    θ = sol.xd[mdl.vehicle.id_θ, :]
    δ = sol.ud[mdl.vehicle.id_δ[1], :]
    pos = sol.xd[mdl.vehicle.id_r, :]
    u_nrml = maximum(T)
    r_span = norm(mdl.traj.r0)
    u_scale = 1/u_nrml*r_span*0.1
    for k = 1:N
        base = pos[1:2, k]
        thrust = -[-T[k]*sin(θ[k]+δ[k]); T[k]*cos(θ[k]+δ[k])]
        tip = base+u_scale*thrust
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#db6245",
                linewidth=1.5,
                solid_capstyle="round",
                zorder=100)
    end

    # ..:: Draw the fuselage ::..
    b_scale = r_span*0.1
    for k = 1:N
        altitude = dot(@k(pos), mdl.env.ey)
        base = @k(pos)
        nose = [-sin(θ[k]); cos(θ[k])]
        tip = base+b_scale*nose
        x = [base[1], tip[1]]
        y = [base[2], tip[2]]
        ax.plot(x, y,
                color="#26415d",
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

#= Plot the velocity trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution. =#
function plot_velocity(mdl::StarshipProblem,
                       sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    traj = mdl.traj
    clr = get_colormap()(1.0)
    t1 = sol.p[veh.id_t1]
    t2 = sol.p[veh.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)
    xy_clrs = ["#db6245", "#5da9a1"]

    fig = create_figure((5, 2.5))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")

    # ..:: Velocity (continuous-time) ::..
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    ct_vel = hcat([
        sample(sol.xc, τ)[mdl.vehicle.id_v] for τ in ct_τ]...)

    for i=1:2
        ax.plot(ct_time, ct_vel[i, :],
                color=xy_clrs[i],
                linewidth=2)
    end

    # ..:: Velocity (discrete-time) ::..
    dt_time = map(τ2t, sol.td)
    dt_vel = sol.xd[mdl.vehicle.id_v, :]
    for i=1:2
        ax.plot(dt_time, dt_vel[i, :],
                linestyle="none",
                marker="o",
                markersize=5,
                markeredgewidth=0,
                markerfacecolor=xy_clrs[i],
                clip_on=false,
                zorder=100)
    end

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

    save_figure("starship_velocity", algo)

    return nothing
end

#= Plot the thrust trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution. =#
function plot_thrust(mdl::StarshipProblem,
                     sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    veh = mdl.vehicle
    traj = mdl.traj
    clr = get_colormap()(1.0)
    t1 = sol.p[veh.id_t1]
    t2 = sol.p[veh.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)
    N = size(sol.xd, 2)
    scale = 1e-6
    y_top = 7.0
    y_bot = 0.0

    fig = create_figure((5, 2.5))
    ax = fig.add_subplot()

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Thrust [MN]")

    # ..:: Thrust bounds ::..
    bnd_min1 = veh.T_min1*scale
    bnd_max1 = veh.T_max1*scale
    bnd_min3 = veh.T_min3*scale
    bnd_max3 = veh.T_max3*scale
    plot_timeseries_bound!(ax, 0.0, t1, bnd_max3, y_top-bnd_max3)
    plot_timeseries_bound!(ax, 0.0, t1, bnd_min3, y_bot-bnd_min3)
    plot_timeseries_bound!(ax, t1, tf, bnd_max1, y_top-bnd_max1)
    plot_timeseries_bound!(ax, t1, tf, bnd_min1, y_bot-bnd_min1)

    # ..:: Thrust value (continuous-time) ::..
    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    ct_thrust = T_RealVector([sample(sol.uc, τ)[mdl.vehicle.id_T]*scale
                              for τ in ct_τ])
    ax.plot(ct_time, ct_thrust,
            color=clr,
            linewidth=2)

    # ..:: Thrust value (discrete-time) ::..
    dt_time = map(τ2t, sol.td)
    dt_thrust = sol.ud[mdl.vehicle.id_T, :]*scale
    ax.plot(dt_time, dt_thrust,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

    save_figure("starship_thrust", algo)

    return nothing
end

#= Plot the gimbal angle trajectory.

Args:
    mdl: the starship problem parameters.
    sol: the trajectory solution. =#
function plot_gimbal(mdl::StarshipProblem,
                     sol::SCPSolution)::Nothing

    # Common values
    algo = sol.algo
    clr = get_colormap()(1.0)
    t1 = sol.p[mdl.vehicle.id_t1]
    t2 = sol.p[mdl.vehicle.id_t2]
    tf = t1+t2
    τs = mdl.traj.τs
    scale = 180/pi
    τ2t = (τ) -> ((τ<=τs) ? τ/τs*t1 : t1+(τ-τs)/(1-τs)*t2)

    fig = create_figure((5, 5))

    ct_res = 500
    ct_τ = T_RealArray(LinRange(0.0, 1.0, ct_res))
    ct_time = map(τ2t, ct_τ)
    dt_time = map(τ2t, sol.td)

    # ..:: Gimbal angle timeseries ::..
    ax = fig.add_subplot(211)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gimbal angle [\$^\\circ\$]")

    # >> Gimbal angle bounds <<
    pad = 2.0
    bnd_max = mdl.vehicle.δ_max*scale
    bnd_min = -mdl.vehicle.δ_max*scale
    y_top = bnd_max+pad
    y_bot = bnd_min-pad
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # >> Delayed gimbal angle (continuous-time) <<
    ct_gimbal_delayed = T_RealVector([
        sample(sol.xc, τ)[mdl.vehicle.id_δd]*scale for τ in ct_τ])
    ax.plot(ct_time, ct_gimbal_delayed,
            color="#db6245",
            linestyle="--",
            linewidth=1,
            dash_capstyle="round")

    # >> Delayed gimbal angle (discrete-time) <<
    dt_gimbal_delayed = sol.xd[mdl.vehicle.id_δd, :]*scale
    ax.plot(dt_time, dt_gimbal_delayed,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0,
            markerfacecolor="#db6245",
            clip_on=false)

    # >> Gimbal angle (continuous-time) <<
    ct_gimbal = T_RealVector([
        sample(sol.uc, τ)[mdl.vehicle.id_δ]*scale for τ in ct_τ])
    ax.plot(ct_time, ct_gimbal,
            color=clr,
            linewidth=2)

    # >> Gimbal angle (discrete-time) <<
    dt_gimbal = sol.ud[mdl.vehicle.id_δ, :]*scale
    ax.plot(dt_time, dt_gimbal,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

    # ..:: Gimbal rate timeseries ::..
    ax = fig.add_subplot(212)

    ax.grid(linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight=true)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Gimbal rate [\$^\\circ\$/s]")

    # >> Gimbal rate bounds <<
    pad = 2.0
    bnd_max = mdl.vehicle.δdot_max*scale
    bnd_min = -mdl.vehicle.δdot_max*scale
    y_top = bnd_max+pad
    y_bot = bnd_min-pad
    plot_timeseries_bound!(ax, 0.0, tf, bnd_max, y_top-bnd_max)
    plot_timeseries_bound!(ax, 0.0, tf, bnd_min, y_bot-bnd_min)

    # >> Actual gimbal rate (discrete-time) <<
    δ = sol.ud[mdl.vehicle.id_δ, 1:end-1]
    δn = sol.ud[mdl.vehicle.id_δ, 2:end]
    dt_δdot = (δn-δ)./diff(dt_time)
    ax.plot(dt_time[2:end], dt_δdot*scale,
            linestyle="none",
            marker="o",
            markersize=5,
            markeredgewidth=0,
            markerfacecolor=clr,
            clip_on=false,
            zorder=100)

    # >> Actual gimbal rate (continuous-time) <<
    δdot = T_ContinuousTimeTrajectory(sol.td[1:end-1], dt_δdot, :zoh)
    ct_δdot = T_RealVector([sample(δdot, τ)*scale for τ in ct_τ])
    ax.plot(ct_time, ct_δdot,
            color=clr,
            linewidth=2,
            zorder=100)

    # >> Constraint (approximate) gimbal rate (discrete-time) <<
    δ = sol.xd[mdl.vehicle.id_δd, :]
    δn = sol.ud[mdl.vehicle.id_δ, :]
    dt_δdot = (δn-δ)./mdl.vehicle.rate_delay
    ax.plot(dt_time, dt_δdot*scale,
            linestyle="none",
            marker="o",
            markersize=3,
            markeredgewidth=0,
            markerfacecolor="#db6245",
            clip_on=false,
            zorder=110)

    # Plot switch time
    _starship__plot_switch_time(ax, t1)

    save_figure("starship_gimbal", algo)

    return nothing
end

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :: Private methods ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

""" Draw the glide slope constraint on an existing plot.

Args:
* `ax`: the figure axis object.
* `mdl`: the starship problem parameters.
* `alt`: (optional) altitude of glide slope "triangle" visualization.
"""
function _starship__plot_glideslope(ax::PyPlot.PyObject,
                                    mdl::StarshipProblem;
                                    alt::T_Real=200.0)::Nothing
    x_gs = alt*tan(mdl.traj.γ_gs)
    ax.plot([-x_gs, 0, x_gs], [alt, 0, alt],
            color="#5da9a1",
            linestyle="--",
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=90)
    return nothing
end

""" Draw the phase switch time on a timeseries plot.

Args:
* `ax`: the figure axis object.
* `t1`: the duration of phase 1.
"""
function _starship__plot_switch_time(ax::PyPlot.PyObject,
                                     t1::T_Real)::Nothing
    ax.axvline(x=t1,
               color="black",
               linestyle="--",
               linewidth=1,
               dash_capstyle="round")
    return nothing
end
