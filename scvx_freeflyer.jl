#= 6-DoF Free-flyer.

Solution via Sequential Convex Programming using the SCvx algorithm. =#

using LinearAlgebra
using JuMP

include("utils/helper.jl")
include("core/problem.jl")
include("core/scvx.jl")
include("models/freeflyer.jl")

###############################################################################
# ..:: Define the trajectory problem data ::..

# >> Free-flyer <<
id_r = 1:3
id_v = 4:6
id_q = 7:10
id_ω = 11:13
id_xt = 14
id_T = 1:3
id_M = 4:6
id_pt = 1
v_max = 0.4
ω_max = deg2rad(10)
T_max = 72e-3
M_max = 2e-3
mass = 7.2
J = diagm([0.1083, 0.1083, 0.1083])
R = sqrt(3)*(0.05/2)
fflyer = FreeFlyerParameters(id_r, id_v, id_q, id_ω, id_xt, id_T, id_M, id_pt,
                             v_max, ω_max, T_max, M_max, mass, J, R)

# >> Trajectory <<
r0 = [7.2; -0.4; 5.0]
v0 = [0.035; 0.035; 0.0]
q0 = T_Quaternion(deg2rad(110), [0.0; 1.0; 1.0]./sqrt(2))
ω0 = zeros(3)
rf = [11.3; 6.0; 4.5]
vf = zeros(3)
qf = T_Quaternion(deg2rad(0), [0.0; 0.0; 1.0])
ωf = zeros(3)
tf_min = 100.0
tf_max = 100.0
traj = TrajectoryParameters(r0, rf, v0, vf, q0, qf, ω0, ωf, tf_min, tf_max)

mdl = FreeFlyerProblem(fflyer, traj)

###############################################################################

###############################################################################
# ..:: Define the trajectory optimization problem ::..

pbm = TrajectoryProblem(mdl)

# Variable dimensions
problem_set_dims!(pbm, 14, 6, 1)

# Initial trajectory guess
problem_set_guess!(pbm,
                   (N, pbm) -> begin
                   veh = pbm.mdl.vehicle
                   traj = pbm.mdl.traj
                   # >> Parameter guess <<
                   p = zeros(pbm.np)
                   flight_time = 0.5*(traj.tf_min+traj.tf_max)
                   p[veh.id_pt] = flight_time
                   # >> State guess <<
                   x = T_RealMatrix(undef, pbm.nx, N)
                   x[veh.id_xt, :] = straightline_interpolate(
                       [flight_time], [flight_time], N)
                   # @ Position/velocity L-shape trajectory @
                   Δτ = flight_time/(N-1)
                   speed = norm(traj.rf-traj.r0, 1)/flight_time
                   times = straightline_interpolate([0.0], [flight_time], N)
                   flight_time_leg = abs.(traj.rf-traj.r0)/speed
                   flight_time_leg_cumul = cumsum(flight_time_leg)
                   r = view(x, veh.id_r, :)
                   v = view(x, veh.id_v, :)
                   for k = 1:N
                   # --- for k
                   tk = @k(times)[1]
                   for i = 1:3
                   # -- for i
                   if tk <= flight_time_leg_cumul[i]
                   # - if tk
                   # Current node is in i-th leg of the trajectory
                   # Endpoint times
                   t0 = (i>1) ? flight_time_leg_cumul[i-1] : 0.0
                   tf = flight_time_leg_cumul[i]
                   # Endpoint positions
                   r0 = copy(traj.r0)
                   r0[1:i-1] = traj.rf[1:i-1]
                   rf = copy(r0)
                   rf[i] = traj.rf[i]
                   @k(r) = linterp(tk, hcat(r0, rf), [t0, tf])
                   # Velocity
                   dir_vec = rf-r0
                   dir_vec /= norm(dir_vec)
                   v_leg = speed*dir_vec
                   @k(v) = v_leg
                   break
                   # - if tk
                   end
                   # -- for i
                   end
                   # --- for k
                   end
                   # @ Quaternion SLERP interpolation @
                   x[veh.id_q, :] = T_RealMatrix(undef, 4, N)
                   for k = 1:N
                   mix = (k-1)/(N-1)
                   @k(view(x, veh.id_q, :)) = vec(slerp_interpolate(
                       traj.q0, traj.qf, mix))
                   end
                   # @ Constant angular velocity @
                   rot_ang, rot_ax = Log(traj.qf*traj.q0')
                   rot_speed = rot_ang/flight_time
                   ang_vel = rot_speed*rot_ax
                   x[veh.id_ω, :] = straightline_interpolate(
                       ang_vel, ang_vel, N)
                   # >> Input guess <<
                   idle = zeros(pbm.nu)
                   u = straightline_interpolate(idle, idle, N)
                   return x, u, p
                   end)

# Cost to be minimized
problem_set_cost!(pbm,
                  # Terminal cost
                  nothing,
                  # Running cost
                  (x, u, p, pbm) -> begin
                  veh = pbm.mdl.vehicle
                  T = u[veh.id_T]
                  M = u[veh.id_M]
                  return T'*T+M'*M
                  end)

# Dynamics constraint
problem_set_dynamics!(pbm,
                      # Dynamics f
                      (τ, x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt] # Time dilation
                      v = x[veh.id_v]
                      q = T_Quaternion(x[veh.id_q])
                      ω = x[veh.id_ω]
                      T = u[veh.id_T]
                      M = u[veh.id_M]
                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = T/veh.m
                      f[veh.id_q] = 0.5*vec(q*T_Quaternion(ω))
                      f[veh.id_ω] = veh.J\(M-cross(ω, veh.J*ω))
                      f *= tdil
                      return f
                      end,
                      # Jacobian df/dx
                      (τ, x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      v = x[veh.id_v]
	              q = T_Quaternion(x[veh.id_q])
	              ω = x[veh.id_ω]
                      dfqdq = 0.5*skew(q*T_Quaternion(ω)*q')
	              dfqdω = 0.5*skew(q)
	              dfωdω = -veh.J\(skew(ω)*J-skew(veh.J*ω))
                      A = zeros(pbm.nx, pbm.nx)
                      A[veh.id_r, veh.id_v] = I(3)
                      A[veh.id_q, veh.id_q] = dfqdq
                      A[veh.id_q, veh.id_ω] = dfqdω[:, 1:3]
	              A[veh.id_ω, veh.id_ω] = dfωdω
                      A *= tdil
                      return A
                      end,
                      # Jacobian df/du
                      (τ, x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      B = zeros(pbm.nx, pbm.nu)
                      B[veh.id_v, veh.id_T] = (1.0/veh.m)*I(3)
                      B[veh.id_ω, veh.id_M] = veh.J\I(3)
                      B *= tdil
                      return B
                      end,
                      # Jacobian df/dp
                      (τ, x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      F = zeros(pbm.nx, pbm.np)
                      F[:, veh.id_pt] = pbm.f(τ, x, u, p)/tdil
                      return F
                      end)

# Convex path constraints on the state
problem_set_X!(pbm, (x, mdl, pbm) -> begin
               traj = pbm.mdl.traj
               veh = pbm.mdl.vehicle
               X = [@constraint(mdl,
                                traj.tf_min <= x[veh.id_xt] <= traj.tf_max)]
               return X
               end)

# Convex path constraints on the input
# TODO

# Nonconvex path inequality constraints
# TODO

# Initial boundary conditions
problem_set_bc!(pbm, :ic,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                tdil = p[veh.id_pt]
                rhs = zeros(pbm.nx)
                rhs[veh.id_r] = traj.r0
                rhs[veh.id_v] = traj.v0
                rhs[veh.id_q] = vec(traj.q0)
                rhs[veh.id_ω] = traj.ω0
                rhs[veh.id_xt] = tdil
                g = x-rhs
                return g
                end,
                # Jacobian dg/dx
                (x, p, pbm) -> begin
                H = I(pbm.nx)
                return H
                end,
                # Jacobian dg/dp
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                K = zeros(pbm.nx, pbm.np)
                K[veh.id_xt, veh.id_pt] = -1.0
                return K
                end)

# Terminal boundary conditions
problem_set_bc!(pbm, :tc,
                # Constraint g
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                traj = pbm.mdl.traj
                tdil = p[veh.id_pt]
                rhs = zeros(pbm.nx)
                rhs[veh.id_r] = traj.rf
                rhs[veh.id_v] = traj.vf
                rhs[veh.id_q] = vec(traj.qf)
                rhs[veh.id_ω] = traj.ωf
                rhs[veh.id_xt] = tdil
                g = x-rhs
                return g
                end,
                # Jacobian dg/dx
                (x, p, pbm) -> begin
                H = I(pbm.nx)
                return H
                end,
                # Jacobian dg/dp
                (x, p, pbm) -> begin
                veh = pbm.mdl.vehicle
                K = zeros(pbm.nx, pbm.np)
                K[veh.id_xt, veh.id_pt] = -1.0
                return K
                end)

###############################################################################

###############################################################################
# ..:: Define the SCvx algorithm parameters ::..

N = 30
Nsub = 15
iter_max = 20
λ = 1e3
ρ_0 = 0.0
ρ_1 = 0.1
ρ_2 = 0.7
β_sh = 2.0
β_gr = 2.0
η_init = 1.0
η_lb = 1e-3
η_ub = 10.0
ε_abs = 0.0#1e-4
ε_rel = 0.1/100
feas_tol = 1e-2
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = SCvxParameters(N, Nsub, iter_max, λ, ρ_0, ρ_1, ρ_2, β_sh, β_gr,
                      η_init, η_lb, η_ub, ε_abs, ε_rel, feas_tol, q_tr,
                      q_exit, solver, solver_options)

###############################################################################

###############################################################################
# ..:: Solve trajectory generation problem using SCvx ::..

scvx_pbm = SCvxProblem(pars, pbm)
sol, history = scvx_solve(scvx_pbm)

###############################################################################

###############################################################################
# ..:: Plot results ::..

plot_trajectory_history(mdl, history)
plot_final_trajectory(mdl, sol)
plot_timeseries(mdl, sol)
plot_convergence(mdl, history)

###############################################################################
