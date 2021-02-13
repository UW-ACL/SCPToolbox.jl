#= Quadrotor Obstacle Avoidance.

Solution via Sequential Convex Programming using the SCvx algorithm. =#

using LinearAlgebra
using ECOS
using Printf
using Plots
using LaTeXStrings
using ColorSchemes

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
tf_min = 2.5
tf_max = 2.5
p_bbox = BoundingBox([tf_min], [tf_max])

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
u_min = [-u_nrm_max; -u_nrm_max; 0.0; u_nrm_min]
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
iter_max = 15
λ = 1e5
ρ_0 = 0.0
ρ_1 = 0.1
ρ_2 = 0.7
β_sh = 2.0
β_gr = 2.0
η_init = 0.5
η_lb = 1e-3
η_ub = 10.0
cvrg_tol = 1e-3
feas_tol = 1e-2
q_tr = Inf
q_exit = Inf
solver = ECOS
solver_options = Dict("verbose"=>0)
pars = SCvxParameters(N, Nsub, iter_max, λ, ρ_0, ρ_1, ρ_2, β_sh, β_gr,
                      η_init, η_lb, η_ub, cvrg_tol, feas_tol, q_tr, q_exit,
                      solver, solver_options)
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
        spbm = SCvxSubproblem(pbm, ref, η)

        add_dynamics!(spbm)
        add_bcs!(spbm)
        add_convex_constraints!(spbm)
        add_nonconvex_constraints!(spbm)
        add_trust_region!(spbm)
        add_cost!(spbm)

        save!(history, spbm)

        # >> Solve the subproblem <<
        solve_subproblem!(spbm)

        # "Emergency exit" the SCvx loop if something bad happened
        # (e.g. numerical problems)
        if unsafe_solution(spbm)
            print_info(history, pbm)
            break
        end

        # >> Check stopping criterion <<
        stop = check_stopping_criterion!(spbm)
        if stop
            print_info(history, pbm)
            break
        end

        # >> Update trust region <<
        try
            ref, η = update_trust_region!(spbm)
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

###############################################################################
# ..:: Plot results ::..

pyplot()

# -----------------------------------------------------------------------------
# >> Trajectory evolution plot through the iterations <<

# Common values
veh = traj_pbm.vehicle
env = traj_pbm.env
num_iter = length(history.subproblems)
font_sz = 10
cmap_offset = 0.1
cmap = cgrad(:thermal; rev = true)

plot(aspect_ratio=:equal,
     xlabel=L"\mathrm{East~position~[m]}",
     ylabel=L"\mathrm{North~position~[m]}",
     tickfontsize=font_sz,
     labelfontsize=font_sz)

# @ Draw the obstacles @
θ = LinRange(0.0, 2*pi, 100)
circle = hcat(cos.(θ), sin.(θ))'
for i = 1:env.obsN
    local H, c = project(get_obstacle(i, traj_pbm)..., [1, 2])
    local vertices = H\circle.+c
    local obs = Shape(vertices[1, :], vertices[2, :])
    plot!(obs;
          reuse=true,
          legend=false,
          seriestype=:shape,
          color="#db6245",
          fillopacity=0.5,
          linewidth=1,
          linecolor="#26415d")
end

# @ Draw the trajectories @
for i = 1:num_iter

    # Extract values for the trajectory at iteration i
    local sol = history.subproblems[i].sol
    local pos = sol.xd[veh.id_r, :]
    local clr = cmap[(i-1)/(num_iter-1)*(1-cmap_offset)+cmap_offset]

    plot!(pos[1, :],
          pos[2, :];
          reuse=true,
          legend=false,
          seriestype=:scatter,
          markershape=:circle,
          markersize=6,
          markerstrokecolor="white",
          markerstrokewidth=0.3,
          color=clr)
end

gui()
# -----------------------------------------------------------------------------

###############################################################################
