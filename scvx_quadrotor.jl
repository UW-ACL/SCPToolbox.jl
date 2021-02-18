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
id_xt = 7
id_u = 1:3
id_σ = 4
id_pt = 1
u_nrm_max = 23.2
u_nrm_min = 0.6
tilt_max = deg2rad(60)
quad = QuadrotorParameters(id_r, id_v, id_xt, id_u, id_σ,
                           id_pt, u_nrm_max, u_nrm_min, tilt_max)

# >> Boundary conditions <<
x0 = zeros(quad.nx)
xf = zeros(quad.nx)
xf[quad.id_r[1:2]] = [2.5; 6.0]
tf_min = 0.0
tf_max = 10.0

traj_pbm = QuadrotorTrajectoryProblem(quad, env, x0, xf, tf_min, tf_max)
###############################################################################

###############################################################################
# ..:: Define the SCvx algorithm parameters ::..
N = 30
Nsub = 15
iter_max = 100
λ = 1e3
ρ_0 = 0.0
ρ_1 = 0.1
ρ_2 = 0.7
β_sh = 2.0
β_gr = 2.0
η_init = 1.0
η_lb = 1e-3
η_ub = 10.0
ε_abs = 1e-3
ε_rel = 1/100
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

    history = SCvxHistory()

    # ..:: Iterate ::..

    for k = 1:pbm.pars.iter_max
        # >> Construct the subproblem <<
        spbm = SCvxSubproblem(pbm, k, η, ref)

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
            print_info(spbm)
            break
        end

        # >> Check stopping criterion <<
        stop = check_stopping_criterion!(spbm)
        if stop
            print_info(spbm)
            break
        end

        # >> Update trust region <<
        try
            ref, η = update_trust_region!(spbm)
        catch e
            isa(e, SCvxError) || rethrow(e)
            print_info(spbm, e)
            break
        end

        # >> Print iteration info <<
        print_info(spbm)
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
