#= Quadrotor Obstacle Avoidance.

Solution via Sequential Convex Programming using the SCvx algorithm. =#

using LinearAlgebra
using ECOS
using Printf

include("src/quadrotor.jl")
include("src/problem.jl")
include("src/scvx.jl")

###############################################################################
# ..:: Define the trajectory generation problem instance ::..

# >> The environment <<
g = 9.81
obsiH = [diagm([2.0; 2.0; 0.0]), diagm([2.0; 2.0; 0.0])]
obsc = [[1.0; 2.0; 0.0], [2.0; 5.0; 0.0]]
env = FlightEnvironmentParameters(g, obsiH, obsc)

# >> The quadrotor <<
id_r = 1:3
id_v = 4:6
id_u = 1:3
id_σ = 4
id_t = 1
u_nrm_max = 23.2
u_nrm_min = 0.6
tilt_max = deg2rad(60)
quad = QuadrotorParameters(id_r, id_v, id_u, id_σ, id_t, u_nrm_max, u_nrm_min,
                           tilt_max, env)

# >> The trajectory bounding boxes <<
t0 = 2.5
tf = 2.5
p_bbox = BoundingBox([t0], [tf])

# Initial
_x = zeros(quad.generic.nx)
_u = fill(NaN, quad.generic.nu)
x_bbox = BoundingBox(_x, _x)
u_bbox = BoundingBox(_u, _u)
init_bbox = XUPBoundingBox(x_bbox, u_bbox, p_bbox)

# Final
_x = zeros(quad.generic.nx)
_x[id_r[1]], _x[id_r[2]] = 2.5, 6.0
_u = fill(NaN, quad.generic.nu)
x_bbox = BoundingBox(_x, _x)
u_bbox = BoundingBox(_u, _u)
trgt_bbox = XUPBoundingBox(x_bbox, u_bbox, p_bbox)

# Path
_x = zeros(quad.generic.nx)
_x[id_r] .= -10.0
_x[id_v] .= -8.0
u_min = [-u_nrm_max; -u_nrm_max; 0.0; 0.0]
u_max = [u_nrm_max; u_nrm_max; u_nrm_max; u_nrm_max]
x_bbox = BoundingBox(_x, -_x)
u_bbox = BoundingBox(u_min, u_max)
path_bbox = XUPBoundingBox(x_bbox, u_bbox, p_bbox)

traj_bbox = TrajectoryBoundingBox(init_bbox, trgt_bbox, path_bbox)

traj_pbm = QuadrotorTrajectoryProblem(quad, env, traj_bbox)
###############################################################################

###############################################################################
# ..:: Define the SCvx algorithm parameters ::..
N = 30
Nsub = 10
iter_max = 30
wvc = 1e4
ρ_0 = 0.0
ρ_1 = 0.1
ρ_2 = 0.7
β_sh = 2.0
β_gr = 2.0
tr_init = 0.5
tr_lb = 0.001
tr_ub = 10.0
cvrg_tol = 1e-3
feas_tol = 1e-2
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = SCvxParameters(N, Nsub, iter_max, wvc, ρ_0, ρ_1, ρ_2, β_sh, β_gr,
                      tr_init, tr_lb, tr_ub, cvrg_tol, feas_tol, solver,
                      solver_options)
###############################################################################

###############################################################################
# ..:: Apply SCvx ::..

#= Apply the SCvx algorithm to solve the trajectory generation problem.

Args:
    pbm: the trajectory problem to be solved.

Returns:
    sol: the SCvx solution structure.
    history: SCvx iteration data history. =#
function scvx_solve(pbm::SCvxProblem)::Tuple{Union{SCvxSolution, Nothing},
                                             SCvxHistory}
    # ..:: Initialize ::..

    η = pbm.pars.η_init
    init_traj = generate_initial_guess(pbm)
    ref = init_traj

    discretize!(init_traj, pbm)

    history = SCvxHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        subpbm = SCvxSubproblem(pbm, ref, η)

        add_dynamics!(subpbm)
        add_bcs!(subpbm)
        add_convex_constraints!(subpbm)
        add_nonconvex_constraints!(subpbm)
        add_trust_region!(subpbm)
        add_cost!(subpbm)

        save!(history, subpbm)

        # >> Solve the subproblem <<
        solve_subproblem!(subpbm)

        # "Emergency exit" the SCvx loop if something bad happened
        # (e.g. numerical problems)
        if unsafe_solution(subpbm)
            print_info(history, pbm)
            break
        end

        # >> Check stopping criterion <<
        stop = check_stopping_criterion!(subpbm)
        if stop
            break
        end

        # >> Update trust region <<
        try
            ref, η = update_trust_region!(subpbm)
        catch e
            print_info(history, pbm, e)
            break
        end

        # >> Print iteration info <<
        print_info(history, pbm)
    end

    # ..:: Save solution ::..
    sol = SCvxSolution(history)

    return sol, history
end

scvx_pbm = SCvxProblem(pars, traj_pbm)

sol, history = scvx_solve(scvx_pbm)
###############################################################################
