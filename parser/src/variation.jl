#= Variational problem of the conic optimization problem.

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

if isdefined(@__MODULE__, :LanguageServer)
    include("program.jl")
end

export jacobian, variation

# ..:: Globals ::..

const ReduceIndexMap = Types.Optional{Dict{Symbol, Vector{Int}}}

# ..:: Methods ::..

"""
    jacobian(x, F[; reduce])

Compute the Jacobian of `F` with respect to `x`, i.e. ``D_x f(x)``. Assuming
that `F` is vector-valued, the Jacobian is a matrix.

# Arguments
- `x`: the subset of the function's arguments to compute the Jacobian with
  respect to.
- `F`: the function itself.

# Keywords
- `reduce`: (optional) an index map to reduce ignore parts of the
  acobian. Suppose that the `i`th element has the value `j`. This means that
  the `i`th atomic argument of the KKT problem corresponds to the `j`th atomic
  argument of the original problem for which the non-reduced (aka "full")
  Jacobian is computed.

# Returns
- `Df`: the Jacobian.
"""
function jacobian(x::Symbol, F::ProgramFunction;
                  reduce::ReduceIndexMap=nothing)::Types.RealMatrix
    # Create Jacobian matrix
    nrows = length(value(F))
    ncols = numel(getfield(program(F), x))
    Df = zeros(nrows, ncols)
    # Indices and arguments
    idx_map_all = function_args_id(F)
    idx_map_arg = function_args_id(F, x)
    args = getfield(F, x)
    # Fill in non-zero blocks
    for (id, jacobian_submatrix) in all_jacobians(F)
        if length(id)==1 && (id in idx_map_arg)
            i = slice_indices(args[idx_map_all[id]])
            Df[:, i] = jacobian_submatrix
        end
    end
    # Reduce if an index map is provided
    if !isnothing(reduce)
        Df = Df[:, reduce[x]]
    end
    return Df
end # function

"""
    jacobian(x, y, F[; reduce])

Compute the Jacobian of `F` with respect to `y` followed by `x`, i.e. ``D_{xy}
f(x, y)``. Assuming that `F` is vector-valued, the Jacobian is a tensor (a 3D
matrix).

# Arguments
- `x`: the part of the function's arguments to compute the Jacobian with
  respect to second.
- `y`: the part of the function's arguments to compute the Jacobian with
  respect to first.
- `F`: the function itself.

# Keywords
- `reduce`: (optional) an index map, see the docstring of `jacobian` above.

# Returns
- `Df`: the Jacobian.
"""
function jacobian(x::Symbol, y::Symbol, F::ProgramFunction;
                  reduce::ReduceIndexMap=nothing)::Types.RealTensor
    # Create Jacobian tensor
    nrows = length(value(F))
    ncols = numel(getfield(program(F), y))
    ndepth = numel(getfield(program(F), x))
    Df = zeros(nrows, ncols, ndepth)
    # Indices and arguments
    symm = x==y
    idx_map_all = function_args_id(F)
    idx_map_x = function_args_id(F, x)
    idx_map_y = function_args_id(F, y)
    xargs, yargs = getfield(F, x), getfield(F, y)
    # Fill in non-zero blocks
    for (id, jacobian_submatrix) in all_jacobians(F)
        if length(id)==2 && ((id[1] in idx_map_x) && (id[2] in idx_map_y))
            i = slice_indices(yargs[idx_map_all[id[2]]])
            j = slice_indices(xargs[idx_map_all[id[1]]])
            Df[:, i, j] = jacobian_submatrix
            if symm && id[1]!=id[2]
                Df[:, i, j] = jacobian_submatrix'
            end
        end
    end
    # Reduce if an index map is provided
    if !isnothing(reduce)
        Df = Df[:, reduce[y], reduce[x]]
    end
    return Df
end # function

"""
    jacobian(x, J[; reduce])

Compute the Jacobian with respect to `x` of the cost function. This just wraps
the `jacobian` functions for `ProgramFunction`, and combines their output
according to the underlying linear combination of cost terms. See the docstring
ofthe corresponding `jacobian` function for `ProgramFunction` for more info.
"""
function jacobian(x::Symbol, J::QuadraticCost;
                  reduce::ReduceIndexMap=nothing)::Types.RealMatrix
    terms = core_terms(J)
    Df_terms = Vector{Types.RealMatrix}(undef, length(terms))
    for i = 1:length(terms)
        Df_terms[i] = terms.a[i]*jacobian(x, terms.f[i]; reduce=reduce)
    end
    Df = sum(Df_terms)
    return Df
end # function

"""
    jacobian(x, y, J)

Compute the Jacobian with respect to `y`, then `x`, of the cost function. This
just wraps the `jacobian` functions for `ProgramFunction`, and combines their
output according to the underlying linear combination of cost terms. See the
docstring ofthe corresponding `jacobian` function for `ProgramFunction` for
more info.
"""
function jacobian(x::Symbol, y::Symbol, J::QuadraticCost;
                  reduce::ReduceIndexMap=nothing)::Types.RealTensor
    terms = core_terms(J)
    Df_terms = Vector{Types.RealTensor}(undef, length(terms))
    for i = 1:length(terms)
        Df_terms[i] = terms.a[i]*jacobian(x, y, terms.f[i]; reduce=reduce)
    end
    Df = sum(Df_terms)
    return Df
end # function

"""
    set_perturbation!(prg, x, dx)

Set the perturbation constraint for the given argument block.

# Arguments
- `prg`: the optimization problem.
- `x`: the argument block.
- `dx`: the corresponding perturbation argument block.
"""
function set_perturbation_constraint!(prg::ConicProgram,
                                      x::ArgumentBlock,
                                      dx::ArgumentBlock)::Nothing

    pert = perturbation(x)
    pert_kind = kind(pert)
    pert_amount = amount(pert)

    for i = 1:length(x)
        ε = pert_amount[i]
        xref = value(x[i])[1]
        δx = dx[i]
        if pert_kind[i]==FIXED
            @add_constraint(
                prg, ZERO, "perturb",
                (δx,), begin
                    local δx = arg[1] #noerr
                    δx[1]-ε
                end)
        elseif pert_kind[i]==ABSOLUTE
            @add_constraint(
                prg, L1, "perturb",
                (δx,), begin
                    local δx = arg[1] #noerr
                    vcat(ε, δx[1])
                end)
        elseif pert_kind[i]==RELATIVE
            abs_xref = abs(xref)
            if abs_xref<=sqrt(eps())
                msg = "The reference value is too small to use a "*
                    "relative perturbation"
                msg *= @sprintf("(%.5e)", abs_xref)
                err = SCPError(0, SCP_BAD_ARGUMENT, msg)
                throw(err)
            end
            @add_constraint(
                prg, L1, "perturb",
                (δx,), begin
                    local δx = arg[1] #noerr
                    vcat(ε*abs_xref, δx[1])
                end)
        end
    end

    return nothing
end # function

"""
    vary!(prg[; perturbation])

Compute the variation of the optimal solution with respect to changes in the
constant arguments of the problem. This sets the appropriate data such that
afterwards the `sensitivity` function can be called for each variable argument.

Internally, this function formulates the linearized KKT optimality conditions
around the optimal solution.

# Arguments
- `prg`: the optimization problem structure.

# Keywords
- `perturbation`: (optional) a fixed perturbation to set for the parameter
  vector.

# Returns
- `bar`: description.
"""
function variation(prg::ConicProgram;
                   ignore_variables::Vector{String}=String[],
                   ignore_constraints::Vector{String}=String[])::Tuple{
                       ArgumentBlockMap, ConicProgram}

    # Initialize the variational problem
    kkt = ConicProgram(solver=prg._solver,
                       solver_options=prg._solver_options)

    # Make the ignore lists and checker function
    varignorelist = [Regex(r) for r in ignore_variables]
    cstignorelist = [Regex(r) for r in ignore_constraints]
    isignored = (obj, ignorelist)->any(occursin.(ignorelist, name(obj)))

    # Create the concatenated primal variable perturbation
    varmap = Dict{ArgumentBlock, Any}()
    idmap = Dict(:x=>Vector{Int}(undef, 0), :p=>Vector{Int}(undef, 0))
    allowed_vars = Dict{Symbol, Vector{ArgumentBlock}}()
    for type in [:x, :p]
        z = getfield(prg, type)
        allowed_vars[type] = Vector{ArgumentBlock}(undef, 0)
        for z_blk in z
            if isignored(z_blk, varignorelist)
                continue
            end

            δz_blk = copy(z_blk, kkt; new_name="δ%s", copyas=VARIABLE)

            # Remove any scaling offset (since perturbations are around zero)
            @scale(δz_blk, dilation(scale(z_blk)))

            # Record in the variable correspondence map
            push!(allowed_vars[type], z_blk)
            append!(idmap[type], slice_indices(z_blk))
            varmap[z_blk] = δz_blk # original -> kkt
            varmap[δz_blk] = z_blk # kkt -> original
        end
    end
    δx_blks = [varmap[z_blk][:] for z_blk in allowed_vars[:x]]
    δp_blks = [varmap[z_blk][:] for z_blk in allowed_vars[:p]]

    # Create the dual variable perturbations
    id_cones_all = 1:length(constraints(prg))
    id_cones_red = Int[]
    λ = dual.(constraints(prg))
    δλ = VariableArgumentBlocks(undef, 0)
    for i in id_cones_all
        C = constraints(prg, i)
        if isignored(C, cstignorelist)
            continue
        end

        push!(id_cones_red, i)
        blk_name = @sprintf("δλ%d", length(id_cones_red))
        push!(δλ, @new_variable(kkt, length(λ[i]), blk_name))
    end
    n_cones_red = length(id_cones_red)

    # Build the constraint function Jacobians
    f = Vector{Types.RealVector}(undef, n_cones_red)
    Dxf = Vector{Types.RealMatrix}(undef, n_cones_red)
    Dpf = Vector{Types.RealMatrix}(undef, n_cones_red)
    Dpxf = Vector{Types.RealTensor}(undef, n_cones_red)
    for i in id_cones_red
        C = constraints(prg, i)
        F = lhs(C)

        # Compute function value and (internally) the Jacobian submatrices
        f[i] = F(jacobians=true)

        # Build the "full" Jacobian with respect to all variables
        Dxf[i] = jacobian(:x, F; reduce=idmap)
        Dpf[i] = jacobian(:p, F; reduce=idmap)
        Dpxf[i] = jacobian(:p, :x, F; reduce=idmap)
    end

    # Build the cost function Jacobians
    J = cost(prg)
    J(jacobians=true)
    DxJ = jacobian(:x, J; reduce=idmap)[1, :]
    DxxJ = jacobian(:x, :x, J; reduce=idmap)[1, :, :]
    DpxJ = jacobian(:p, :x, J; reduce=idmap)[1, :, :]

    # Check that complementary slackness holds at the reference solution
    max_viol = -Inf
    for i = 1:n_cones_red
        compl_slack = dot(f[i], λ[i])
        if abs(compl_slack)>max_viol
            max_viol = abs(compl_slack)
        end
        # warn = (abs(compl_slack)>1e-10) ? "*** " : ""
        # @printf("%scompl. slack. %d = %.4e\n", warn, i, compl_slack)
    end
    @printf("Complementary slackness violation = %.4e\n", max_viol)

    # Check that stationarity holds at the reference solution
    stat = DxJ-sum(Dxf[i]'*λ[i] for i=1:n_cones_red)
    max_viol = norm(stat, Inf)
    @printf("Stationarity violation = %.4e\n", max_viol)
    # for i = 1:length(stat)
    #     warn = (abs(stat[i])>1e-10) ? "*** " : ""
    #     @printf("%sstationarity %s = %.4e\n", warn, i, stat[i])
    # end

    # Primal feasibility
    num_x_blk = length(δx_blks)
    num_p_blk = length(δp_blks)
    num_xp_blk = num_x_blk+num_p_blk
    idcs_x = 1:num_x_blk
    idcs_p = (1:num_p_blk).+idcs_x[end]
    for i = 1:n_cones_red
        C = constraints(prg, i)
        K = kind(C)
        @add_constraint(
            kkt, K, "primal_feas",
            (δx_blks..., δp_blks...),
            begin
                local δx = vcat(arg[idcs_x]...) #noerr
                local δp = vcat(arg[idcs_p]...) #noerr
                f[i]+Dxf[i]*δx+Dpf[i]*δp
            end)
    end

    # Dual feasibility
    for i = 1:n_cones_red
        K = kind(constraints(prg, i))
        @add_constraint(
            kkt, dual(K), "dual_feas",
            (δλ[i],),
            begin
                local δλ = arg[1] #noerr
                λ[i]+δλ
            end)
    end

    # Complementary slackness
    ρcs = @new_variable(kkt, n_cones_red, "ρcs")
    for i = 1:n_cones_red
        # @add_constraint(
        #     kkt, ZERO, "compl_slack",
        #     (δx_blks..., δp_blks..., δλ[i]),
        #     begin
        #         local δx = vcat(arg[idcs_x]...) #noerr
        #         local δp = vcat(arg[idcs_p]...) #noerr
        #         local δλ = arg[num_xp_blk+1] #noerr
        #         dot(f[i], δλ)+dot(Dxf[i]*δx+Dpf[i]*δp, λ[i])
        #     end)
        @add_constraint(
            kkt, L1, "compl_slack",
            (δx_blks..., δp_blks..., δλ[i], ρcs[i]),
            begin
                local δx = vcat(arg[idcs_x]...) #noerr
                local δp = vcat(arg[idcs_p]...) #noerr
                local δλ = arg[num_xp_blk+1] #noerr
                local ρcs = arg[num_xp_blk+2] #noerr
                vcat(ρcs, dot(f[i], δλ)+dot(Dxf[i]*δx+Dpf[i]*δp, λ[i]))
            end)
    end

    # Stationarity
    np = numel(prg.p)
    @add_constraint(
        kkt, ZERO, "stat",
        (δx_blks..., δp_blks..., δλ...),
        begin
            local δx = vcat(arg[idcs_x]...) #noerr
            local δp = vcat(arg[idcs_p]...) #noerr
            local δλ = arg[(1:n_cones_red).+num_xp_blk] #noerr
            local ∇L = DxxJ*δx+DpxJ*δp
            if np>0
                local Dxf_vary_p = (i)->sum(Dpxf[i][:, :, j]*δp[j] for j=1:np)
                for i = 1:n_cones_red
                    ∇L -= Dxf_vary_p(i)'*λ[i]+Dxf[i]'*δλ[i]
                end
            else
                for i = 1:n_cones_red
                    ∇L -= Dxf[i]'*δλ[i]
                end
            end
            ∇L
        end)

    # ρst = @new_variable(kkt, "ρst")
    # @add_constraint(
    #     kkt, LINF, "stat",
    #     (δx_blks..., δp_blks..., δλ..., ρst),
    #     begin
    #         local δx = vcat(arg[idcs_x]...) #noerr
    #         local δp = vcat(arg[idcs_p]...) #noerr
    #         local δλ = arg[(1:n_cones_red).+num_xp_blk] #noerr
    #         local ρst = arg[end] #noerr
    #         local ∇L = DxxJ*δx+DpxJ*δp
    #         if np>0
    #             local Dxf_vary_p = (i)->sum(Dpxf[i][:, :, j]*δp[j] for j=1:np)
    #             for i = 1:n_cones_red
    #                 ∇L -= Dxf_vary_p(i)'*λ[i]+Dxf[i]'*δλ[i]
    #             end
    #         else
    #             for i = 1:n_cones_red
    #                 ∇L -= Dxf[i]'*δλ[i]
    #             end
    #         end
    #         vcat(ρst, ∇L)
    #     end)

    # Set the perturbation constraints
    for kind in [:x, :p] #noinfo
        z_blks = allowed_vars[kind]
        for z_blk in z_blks
            δz_blk = varmap[z_blk]
            for i=1:length(δz_blk)
                z, δz = z_blk[i], δz_blk[i]
                set_perturbation_constraint!(kkt, z, δz)
            end
        end
    end

    # Set a cost to minimize complementary slackness relaxation
    # @add_cost(kkt, (ρst,), begin
    #               local ρst, = arg #noerr
    #               ρst[1]
    #           end)
    # @add_cost(kkt, (ρcs, ρst), begin
    #               local ρcs, ρst = arg #noerr
    #               sum(ρcs)+ρst[1]
    #           end)
    @add_cost(kkt, (ρcs,), begin
                  local ρcs, = arg #noerr
                  sum(ρcs)
              end)

    return varmap, kkt
end # function
