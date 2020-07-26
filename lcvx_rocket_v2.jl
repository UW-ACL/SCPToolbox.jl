# LCvx: 3-DoF Fuel-Optimal Rocket Landing

using LinearAlgebra
using Convex
using ECOS
using PyPlot
using Printf

################################################################################
# ..:: Data structures ::..
#
global const LCvxReal = Float64
global const LCvxVector = Vector{LCvxReal}
global const LCvxMatrix = Matrix{LCvxReal}
#
module Data
#
global const LCvxReal = Float64
global const LCvxVector = Vector{LCvxReal}
global const LCvxMatrix = Matrix{LCvxReal}
#
mutable struct Rocket
    g::LCvxVector # [m/s²] Acceleration due to gravity
    ω::LCvxVector # [rad/s] Planet angular velocity
    m_dry::LCvxReal # [kg] Dry mass (structure)
    m_wet::LCvxReal # [kg] Wet mass (structure+fuel)
    Isp::LCvxReal # [s] Specific impulse
    ϕ::LCvxReal # [rad] Rocket engine cant angle
    α::LCvxReal # [s/m] 1/(rocket engine exit velocity)
    ρ_min::LCvxReal # [N] Minimum thrust
    ρ_max::LCvxReal # [N] Maximum thrust
    γ_gs::LCvxReal # [rad] Maximum approach angle
    γ_p::LCvxReal # [rad] Maximum pointing angle
    v_max::LCvxReal # [m/s] Maximum velocity
    r0::LCvxVector # [m] Initial position
    v0::LCvxVector # [m] Initial velocity
    Δt::LCvxReal # [s] Discretization time step
    A_c::LCvxMatrix # Continuous-time dynamics A matrix
    B_c::LCvxMatrix # Continuous-time dynamics B matrix
    p_c::LCvxVector # Continuous-time dynamics p vector
    n::Int # Number of states
    m::Int # Number of inputs
end
#
end
#
function Rocket(g::LCvxVector,ω::LCvxVector,m_dry::LCvxReal,
                m_wet::LCvxReal,Isp::LCvxReal,ϕ::LCvxReal,ρ_min::LCvxReal,
                ρ_max::LCvxReal,γ_gs::LCvxReal,γ_p::LCvxReal,
                v_max::LCvxReal,r0::LCvxVector,v0::LCvxVector,
                Δt::LCvxReal)::Data.Rocket
    ############################################################################
    # ROCKET initializes the rocket object.
    #
    # Parameters
    # ----------
    # See Data.Rocket struct definition
    # 
    # Returns
    # -------
    # rocket::Data.Rocket = rocket object, with empty discrete dynamics {A,B,p}.
    ############################################################################
    # >> Continuous-time dynamics <<
    gₑ = 9.807 # [m/s²] Standard gravity
    α = 1/(Isp*gₑ*cos(ϕ))
    ω_x = LCvxMatrix([0 -ω[3] ω[2];ω[3] 0 -ω[1];-ω[2] ω[1] 0])
    A_c = LCvxMatrix([zeros(3,3) I(3) zeros(3);
                      -(ω_x)^2 -2*ω_x zeros(3);
                      zeros(1,7)])
    B_c = LCvxMatrix([zeros(3,4);
                      I(3) zeros(3,1);
                      zeros(1,3) -α])
    p_c = LCvxVector(vcat(zeros(3),g,0))
    n,m = size(B_c)
    # >> Make rocket object <<
    rocket = Data.Rocket(g,ω,m_dry,m_wet,Isp,ϕ,α,ρ_min,ρ_max,γ_gs,γ_p,v_max,
                         r0,v0,Δt,A_c,B_c,p_c,n,m)
    return rocket
end
################################################################################

################################################################################
# ..:: ZOH discretization ::..
function c2d(rocket::Data.Rocket,Δt::LCvxReal)::Tuple{LCvxMatrix,LCvxMatrix,
                                                      LCvxVector}
    ############################################################################
    # C2D Discretize rocket dynamics at Δt time step using zeroth-order hold
    # (ZOH). This updates the {A,B,p} member variables of the rocket object.
    #
    # Parameters
    # ----------
    # rocket::Data.Rocket = the rocket object.
    # Δt::LCvxReal = the discrete time step.
    ############################################################################
    A_c,B_c,p_c,n,m = rocket.A_c,rocket.B_c,rocket.p_c,rocket.n,rocket.m
    _M = exp(LCvxMatrix([A_c B_c p_c;zeros(m+1,n+m+1)])*Δt)
    A = _M[1:n,1:n]
    B = _M[1:n,n+1:n+m]
    p = _M[1:n,n+m+1]
    return (A,B,p)
end
################################################################################

################################################################################
# ..:: Parameters ::..
e_x = LCvxVector([1,0,0])
e_y = LCvxVector([0,1,0])
e_z = LCvxVector([0,0,1])
g = -3.7114*e_z
θ = 30*π/180 # [rad] Latitude of landing site
T_sidereal_mars = 24.6229*3600 # [s]
ω = (2π/T_sidereal_mars)*(e_x*cos(θ)+e_y*0+e_z*sin(θ))
m_dry = 1505.0
m_wet = 1905.0
Isp = 225.0
n_eng = 6 # Number of engines
ϕ = 27*π/180 # [rad] Engine cant angle off vertical
T_max = 3.1e3 # [N] Max physical thrust of single engine
T_1 = 0.3*T_max # [N] Min allowed thrust of single engine
T_2 = 0.8*T_max # [N] Max allowed thrust of single engine
ρ_min = n_eng*T_1*cos(ϕ)
ρ_max = n_eng*T_2*cos(ϕ)
γ_gs = 86*π/180
γ_p = 40*π/180
v_max = 800*1e3/3600
r0 = (2*e_x+0*e_y+1.5*e_z)*1e3
v0 = 80*e_x+0*e_y-75*e_z
Δt = 1e0
rocket = Rocket(g,ω,m_dry,m_wet,Isp,ϕ,ρ_min,ρ_max,γ_gs,γ_p,v_max,r0,v0,Δt)
################################################################################

################################################################################
# ..:: Golden search ::..
function golden(f::Function,a::LCvxReal,b::LCvxReal,
                tol::LCvxReal)::Tuple{LCvxReal,LCvxReal}
    ############################################################################
    # BISECTION golden search for minimizing a unimodal function f(x) on the
    # interval [a,b] to within a prescribed golerance in x. Implementation is
    # based on [1].
    #
    # [1] M. J. Kochenderfer and T. A. Wheeler, Algorithms for
    # Optimization. Cambridge, Massachusetts: The MIT Press, 2019.
    #
    # Parameters
    # ----------
    # f::Function   = oracle with call signature v=f(x) where v::LCvxReal
    #                 The value v is saught to be minimized.
    # a::LCvxReal   = search domain lower bound.
    # b::LCvxReal   = search domain upper bound.
    # tol::LCvxReal = tolerance in terms of maximum distance that the minimizer
    #                 x∈[a,b] is away from a or b.
    #
    # Returns
    # -------
    # sol::Tuple{LCvxReal,LCvxReal} = structure where s[1] is the argmin and
    #                                 s[2] is the argmax.
    ############################################################################
    ϕ = (1+√5)/2
    n = ceil(log((b-a)/tol)/log(ϕ)+1)
    ρ = ϕ-1
    d = ρ*b+(1-ρ)*a
    yd = f(d)
    for i = 1:n-1
        c = ρ*a+(1-ρ)*b
        yc = f(c)
        if yc<yd
            b,d,yd = d,c,yc
        else
            a,b = b,c
        end
    end
    x_sol = (a+b)/2
    sol = (x_sol,f(x_sol))
    return sol
end
################################################################################

################################################################################
# ..:: Solve fixed-final time optimization problem ::..
function solve_pdg_fft(rocket::Data.Rocket,t_f::LCvxReal)
    # >> Discretize [0,t_f] interval <<
    # If t_f does not divide into rocket.Δt intervals evenly, then reduce Δt by
    # minimum amount to get an integer number of intervals
    N = Int(floor(t_f/rocket.Δt))+1+Int(t_f%rocket.Δt!=0) # Number of time nodes
    Δt = t_f/(N-1)
    t = LCvxVector(0.0:Δt:t_f)
    A,B,p = c2d(rocket,Δt)
    # >> (Scaled) variables <<
    X_s = Variable(7,N)
    U_s = Variable(4,N-1)
    # >> Scaling (for better numerical behaviour) <<
    # @ Scaling matrices @
    #
    s_r = zeros(3)
    S_r = Diagonal([max(1.0,abs(rocket.r0[i])) for i=1:3])
    #
    s_v = zeros(3)
    S_v = Diagonal([max(1.0,abs(rocket.v0[i])) for i=1:3])
    #
    s_z = (log(rocket.m_dry)+log(rocket.m_wet))/2
    S_z = log(rocket.m_wet)-s_z
    #
    s_u = LCvxVector([0,0,0.5*(rocket.ρ_min/rocket.m_wet*cos(rocket.γ_p)+
                               rocket.ρ_max/rocket.m_dry)])
    S_u = Diagonal([rocket.ρ_max/rocket.m_dry*sin(rocket.γ_p),
                    rocket.ρ_max/rocket.m_dry*sin(rocket.γ_p),
                    rocket.ρ_max/rocket.m_dry-s_u[3]])
    #
    s_ξ,S_ξ = s_u[3],S_u[3,3]
    # @ Unscaled variables @
    #    
    X = [S_r*X_s[1:3,:]+repeat(s_r,1,N);
         S_v*X_s[4:6,:]+repeat(s_v,1,N);
         S_z*X_s[7,:]+repeat([s_z],1,N)]
    U = [S_u*U_s[1:3,:]+repeat(s_u,1,N-1);
         S_ξ*U_s[4,:]+repeat([s_ξ],1,N-1)]
    #
    r = X[1:3,:]
    v = X[4:6,:]
    z = X[7,:]
    u = U[1:3,:]
    ξ = U[4,:]
    # >> Cost function <<
    objective = Δt*sum(ξ[k] for k=1:N-1)
    # >> Constraints <<
    # @ Dynamics @
    constraints = Constraint[X[:,k+1]==A*X[:,k]+B*U[:,k]+p for k=1:N-1]
    # @ Thrust bounds (approximate) @
    z0 = (k) -> log(rocket.m_wet-rocket.α*rocket.ρ_max*t[k])
    μ_min = (k) -> rocket.ρ_min*exp(-z0(k))
    μ_max = (k) -> rocket.ρ_max*exp(-z0(k))
    δz = (k) -> z[k]-z0(k)
    for k = 1:N-1
        push!(constraints,ξ[k]>=μ_min(k)*(1-δz(k)+0.5*square(δz(k))))
        push!(constraints,ξ[k]<=μ_max(k)*(1-δz(k)))
    end
    # @ Mass physical bounds constraint @
    for k = 1:N
        push!(constraints,z0(k)<=z[k])
        push!(constraints,z[k]<=log(rocket.m_wet-rocket.α*rocket.ρ_min*t[k]))
    end
    # @ Thrust bounds LCvx @
    for k = 1:N-1
        push!(constraints,norm(u[:,k],2)<=ξ[k])
    end
    # @ Attitude pointing constraint @
    e_z = LCvxVector([0,0,1])
    for k = 1:N-1
        push!(constraints,dot(u[:,k],e_z)>=ξ[k]*cos(rocket.γ_p))
    end
    # # @ Glide slope constraint @
    _n1 = LCvxVector([cos(rocket.γ_gs),0,-sin(rocket.γ_gs)])
    _n2 = LCvxVector([0,cos(rocket.γ_gs),-sin(rocket.γ_gs)])
    _n3 = LCvxVector([-cos(rocket.γ_gs),0,-sin(rocket.γ_gs)])
    _n4 = LCvxVector([0,-cos(rocket.γ_gs),-sin(rocket.γ_gs)])
    H_gs = transpose(hcat(_n1,_n2,_n3,_n4))
    h_gs = zeros(4)
    for k = 1:N
        push!(constraints,H_gs*r[:,k]<=h_gs)
    end
    # @ Velocity upper bound @
    for k = 1:N
        push!(constraints,norm(v[:,k],2)<=rocket.v_max)
    end
    # @ Boundary conditions @
    push!(constraints,r[:,1]==rocket.r0)
    push!(constraints,v[:,1]==rocket.v0)
    push!(constraints,z[1]==log(rocket.m_wet))
    push!(constraints,r[:,N]==zeros(3))
    push!(constraints,v[:,N]==zeros(3))
    push!(constraints,z[N]>=log(rocket.m_dry))
    # >> Solve problem <<
    problem = minimize(objective,constraints)
    solve!(problem, ECOSSolver())
    # >> Extract solution <<
    r = evaluate(r)
    v = evaluate(v)
    z = evaluate(z)[1,:]
    u = evaluate(u)
    ξ = evaluate(ξ)[1,:]
    return (t,r,v,z,u,ξ)
end
#
opti_t,opti_r,opti_v,opti_z,opti_u,opti_ξ = solve_pdg_fft(rocket,85.)
################################################################################

################################################################################
# ..:: Simulate ::..
function rk4(f::Function,x0::LCvxVector,
             Δt::LCvxReal,T::LCvxReal)::Tuple{LCvxVector,LCvxMatrix}
    # >> Make time grid <<
    t = LCvxVector(0.0:Δt:T)
    if (T-t[end])>=√eps()
        push!(t,T)
    end
    N = length(t)
    # >> Initialize <<
    X = LCvxMatrix(undef,length(x0),N)
    X[:,1] = x0
    # >> Integrate <<
    for n = 1:N-1
        y = X[:,n]
        h = t[n+1]-t[n]
        t_ = t[n]
        k1 = f(t_,y)
        k2 = f(t_+h/2,y+h*k1/2)
        k3 = f(t_+h/2,y+h*k2/2)
        k4 = f(t_+h,y+h*k3)
        X[:,n+1] = y+h/6*(k1+2*k2+2*k3+k4)
    end
    return (t,X)
end
#
function simulate(rocket::Data.Rocket,control::Function,
                  t_f::LCvxReal)::Tuple{LCvxVector,LCvxMatrix,LCvxMatrix,
                                        LCvxVector,LCvxMatrix}
    dynamics = (t,x) -> rocket.A_c*x+rocket.B_c*control(t,x,rocket)+rocket.p_c
    x0 = LCvxVector(vcat(rocket.r0,rocket.v0,log(rocket.m_wet)))
    Δt = 1e-2
    t,X = rk4(dynamics,x0,Δt,t_f)
    U = LCvxMatrix(hcat([control(t[n],X[:,n],rocket) for n = 1:length(t)]...))
    r = X[1:3,:] # [m] Position
    v = X[4:6,:] # [m/s] Velocity
    m = exp.(X[7,:]) # [kg] Mass
    T = LCvxMatrix(transpose(hcat([m.*U[i,:] for i=1:3]...))) # [N] Thrust
    return (t,r,v,m,T)
end
################################################################################

################################################################################
# ..:: Test out "hover" control law ::..
function hover_controller(t::LCvxReal,x::LCvxVector,
                          rocket::Data.Rocket)::LCvxVector
    z = x[7]
    m = exp.(z)
    e_z = LCvxVector([0,0,1])
    g = norm(rocket.g,2)
    T = m*g*e_z
    u = LCvxVector(vcat(T/m,norm(T,2)/m))
    return u
end
#
# t_f = 10.0 # [s] Simulation time
# t,p,v,m,T = simulate(rocket,hover_controller,t_f)
################################################################################

################################################################################
# ..:: Simulate optimal control law ::..
function optimal_controller(t::LCvxReal,x::LCvxVector,
                            rocket::Data.Rocket,
                            opti_t::LCvxVector,opti_u::LCvxMatrix)::LCvxVector
    # >> Get current mass <<
    z = x[7]
    m = exp.(z)
    # >> Get current optimal acceleration (ZOH interpolation) <<
    i = findlast(τ->τ<=t,opti_t)
    if typeof(i)==Nothing || i>=size(opti_u,2)
        u = opti_u[:,end]
    else
        u = opti_u[:,i]
    end
    # >> Get current optimal thrust <<
    T = u*m
    # >> Create the input vector <<
    u = LCvxVector(vcat(T/m,norm(T,2)/m))
    return u
end
#
optimal_control = (t,x,rocket) -> optimal_controller(t,x,rocket,opti_t,opti_u)
t_f = opti_t[end] # [s] Simulation time
t,r,v,m,T = simulate(rocket,optimal_control,t_f)
################################################################################

################################################################################
# ..:: Plot ::..
# >> Position <<
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(t,r[1,:],color="red",label="x")
ax.plot(opti_t,opti_r[1,:],color="red",linestyle="none",marker=".",
        markersize=5)
ax.plot(t,r[2,:],color="green",label="y")
ax.plot(opti_t,opti_r[2,:],color="green",linestyle="none",marker=".",
        markersize=5)
ax.plot(t,r[3,:],color="blue",label="z")
ax.plot(opti_t,opti_r[3,:],color="blue",linestyle="none",marker=".",
        markersize=5)
plt.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Position [m]")
# >> Velocity <<
fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(t,v[1,:],color="red",label="x")
ax.plot(opti_t,opti_v[1,:],color="red",linestyle="none",marker=".",
        markersize=5)
ax.plot(t,v[2,:],color="green",label="y")
ax.plot(opti_t,opti_v[2,:],color="green",linestyle="none",marker=".",
        markersize=5)
ax.plot(t,v[3,:],color="blue",label="z")
ax.plot(opti_t,opti_v[3,:],color="blue",linestyle="none",marker=".",
        markersize=5)
plt.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Velocity [m/s]")
# >> Mass <<
opti_m = exp.(opti_z)
#
fig = plt.figure(3)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(t,m,color="black")
ax.plot(opti_t,opti_m,color="black",linestyle="none",marker=".",
        markersize=5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Mass [kg]")
# >> Thrust <<
opti_T = LCvxMatrix(transpose(hcat([opti_m[1:end-1].*opti_u[i,:] for i=1:3]...)))
#
fig = plt.figure(4)
plt.clf()
ax = fig.add_subplot(111)
ax.plot(t,T[1,:],color="red",label="x")
ax.plot(opti_t[1:end-1],opti_T[1,:],color="red",linestyle="none",marker=".",
        markersize=5)
ax.plot(t,T[2,:],color="green",label="y")
ax.plot(opti_t[1:end-1],opti_T[2,:],color="green",linestyle="none",marker=".",
        markersize=5)
ax.plot(t,T[3,:],color="blue",label="z")
ax.plot(opti_t[1:end-1],opti_T[3,:],color="blue",linestyle="none",marker=".",
        markersize=5)
ax.axhline(y=rocket.ρ_min,color="black",linestyle="--",label="ρ_min")
ax.axhline(y=rocket.ρ_max,color="black",linestyle="--",label="ρ_max")
ax.plot(t,m*norm(rocket.g,2),color="gray",linestyle="--",label="hover")
plt.legend()
ax.set_xlabel("Time [s]")
ax.set_ylabel("Thrust [N]")
################################################################################
