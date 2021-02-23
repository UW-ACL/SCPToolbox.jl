#= Quadrotor Obstacle Avoidance.

Solution via Sequential Convex Programming using the SCvx algorithm. =#

using LinearAlgebra
using JuMP
using Plots
using LaTeXStrings
using ColorSchemes

include("utils/helper.jl")
include("core/problem.jl")
include("core/scvx.jl")
include("models/quadrotor.jl")

###############################################################################
# ..:: Define the trajectory problem data ::..

# >> Quadrotor <<
id_r = 1:3
id_v = 4:6
id_xt = 7
id_u = 1:3
id_σ = 4
id_pt = 1
u_max = 23.2
u_min = 0.6
tilt_max = deg2rad(60)
quad = QuadrotorParameters(id_r, id_v, id_xt, id_u, id_σ,
                           id_pt, u_max, u_min, tilt_max)

# >> Environment <<
g = 9.81
obsiH = [diagm([2.0; 2.0; 0.0]),
         diagm([1.5; 1.5; 0.0])]
obsc = [[1.0; 2.0; 0.0],
        [2.0; 5.0; 0.0]]
env = EnvironmentParameters(g, obsiH, obsc)

# >> Trajectory <<
r0 = zeros(3)
rf = zeros(3)
rf[1:2] = [2.5; 6.0]
v0 = zeros(3)
vf = zeros(3)
tf_min = 0.0
tf_max = 2.5
traj = TrajectoryParameters(r0, rf, v0, vf, tf_min, tf_max)

mdl = QuadrotorProblem(quad, env, traj)

###############################################################################

###############################################################################
# ..:: Define the trajectory optimization problem ::..

pbm = TrajectoryProblem(mdl)

# Variable dimensions
problem_set_dims!(pbm, 7, 4, 1)

# Initial trajectory guess
problem_set_guess!(pbm,
                   (N, pbm) -> begin
                   veh = pbm.mdl.vehicle
                   traj = pbm.mdl.traj
                   g = pbm.mdl.env.g
                   # Parameter guess
                   p = zeros(pbm.np)
                   p[veh.id_pt] = 0.5*(traj.tf_min+traj.tf_max)
                   # State guess
                   x0 = zeros(pbm.nx)
                   xf = zeros(pbm.nx)
                   x0[veh.id_r] = traj.r0
                   xf[veh.id_r] = traj.rf
                   x0[veh.id_v] = traj.v0
                   xf[veh.id_v] = traj.vf
                   x0[veh.id_xt] = p[veh.id_pt]
                   xf[veh.id_xt] = p[veh.id_pt]
                   x = straightline_interpolate(x0, xf, N)
                   # Input guess
                   hover = [-g; norm(g)]
                   u = straightline_interpolate(hover, hover, N)
                   return x, u, p
                   end)

# Cost to be minimized
problem_set_cost!(pbm,
                  # Terminal cost
                  nothing,
                  # Running cost
                  (x, u, p, pbm) -> begin
                  σ = u[pbm.mdl.vehicle.id_σ]
                  return σ^2
                  end)

# Dynamics constraint
problem_set_dynamics!(pbm,
                      # Dynamics f
                      (τ, x, u, p, pbm) -> begin
                      g = pbm.mdl.env.g
                      veh = pbm.mdl.vehicle
                      v = x[veh.id_v]
                      uu = u[veh.id_u]
                      tdil = p[veh.id_pt] # Time dilation
                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = uu+g
                      f *= tdil
                      return f
                      end,
                      # Jacobian df/dx
                      (τ, x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      A = zeros(pbm.nx, pbm.nx)
                      A[veh.id_r, veh.id_v] = I(3)
                      A *= tdil
                      return A
                      end,
                      # Jacobian df/du
                      (τ, x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      tdil = p[veh.id_pt]
                      B = zeros(pbm.nx, pbm.nu)
                      B[veh.id_v, veh.id_u] = I(3)
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
problem_set_U!(pbm, (u, mdl, pbm) -> begin
               veh = pbm.mdl.vehicle
               uu = u[veh.id_u]
               σ = u[veh.id_σ]
               U = [@constraint(mdl, veh.u_min <= σ);
                    @constraint(mdl, σ <= veh.u_max);
                    @constraint(mdl, vcat(σ, uu) in
                                MOI.SecondOrderCone(pbm.nu));
                    @constraint(mdl, σ*cos(veh.tilt_max) <= uu[3])]
               return U
               end)

# Nonconvex path inequality constraints
problem_set_s!(pbm,
               # Constraint s
               (x, u, p, pbm) -> begin
               env = pbm.mdl.env
               veh = pbm.mdl.vehicle
               s = zeros(env.obsN)
               for i = 1:env.obsN
               H, c = get_obstacle(i, pbm.mdl)
               r = x[veh.id_r]
               s[i] = 1-norm(H*(r-c))
               end
               return s
               end,
               # Jacobian ds/dx
               (x, u, p, pbm) -> begin
               env = pbm.mdl.env
               veh = pbm.mdl.vehicle
               C = zeros(env.obsN, pbm.nx)
               for i = 1:env.obsN
               H, c = get_obstacle(i, pbm.mdl)
               r = x[veh.id_r]
               C[i, veh.id_r] = -(r-c)'*(H'*H)/norm(H*(r-c))
               end
               return C
               end,
               # Jacobian ds/du
               (x, u, p, pbm) -> begin
               env = pbm.mdl.env
               D = zeros(env.obsN, pbm.nu)
               return D
               end,
               # Jacobian ds/dp
               (x, u, p, pbm) -> begin
               env = pbm.mdl.env
               G = zeros(env.obsN, pbm.np)
               return G
               end)

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
ε_abs = 1e-4
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
plot_input_norm(mdl, sol)
plot_tilt_angle(mdl, sol)
plot_convergence(mdl, history)

###############################################################################
