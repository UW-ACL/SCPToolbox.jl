#= 6-DoF Free-flyer.

Solution via Sequential Convex Programming using the SCvx algorithm. =#

using LinearAlgebra

include("core/problem.jl")
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
ω_max = 1.0
T_max = 72e-3
M_max = 2e-3
mass = 7.2
J = diagm([0.1083, 0.1083, 0.1083])
R = sqrt(3)*(0.05/2)
fflyer = FreeFlyerParameters(id_r, id_v, id_q, id_ω, id_xt, id_T, id_M, id_pt,
                             T_max, M_max, mass, J, R)

mdl = FreeFlyerProblem(fflyer)

###############################################################################

###############################################################################
# ..:: Define the trajectory optimization problem ::..

pbm = TrajectoryProblem(mdl)

# Variable dimensions
problem_set_dims!(pbm, 14, 6, 1)

# Initial trajectory guess
# TODO

# Cost to be minimized
# TODO

# Dynamics constraint
problem_set_dynamics!(pbm,
                      # Dynamics f
                      (τ, x, u, p, pbm) -> begin
                      veh = pbm.mdl.vehicle
                      v = x[veh.id_v]
                      q = T_Quaternion(x[veh.id_q])
                      ω = x[veh.id_ω]
                      T = u[veh.id_T]
                      M = u[veh.id_M]
                      tdil = p[veh.id_pt] # Time dilation
                      f = zeros(pbm.nx)
                      f[veh.id_r] = v
                      f[veh.id_v] = T/veh.m
                      f[veh.id_q] = 0.5*vec(q*T_Quaternion(ω))
                      f[veh.id_ω] = veh.J\(M-cross(ω, J*ω))
                      f *= tdil
                      return f
                      end,
                      # Jacobian df/dx
                      (τ, x, u, p, pbm) -> begin
                      # TODO
                      end,
                      # Jacobian df/du
                      (τ, x, u, p, pbm) -> begin
                      # TODO
                      end,
                      # Jacobian df/dp
                      (τ, x, u, p, pbm) -> begin
                      # TODO
                      end)

# Convex path constraints on the state
# TODO

# Convex path constraints on the input
# TODO

# Nonconvex path inequality constraints
# TODO

# Initial boundary conditions
# TODO

# Terminal boundary conditions
# TODO

###############################################################################
