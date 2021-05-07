#= General conic linear (convex) optimization problem.

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
    include("general.jl")
    include("cone.jl")
    include("argument.jl")
    include("function.jl")
    include("constraint.jl")
    include("cost.jl")
end

export ConicProgram, numel, value, dual, constraints, name, cost, solve!,
    vary!
export @new_variable, @new_parameter, @add_constraint,
    @set_cost, @set_feasibility

# ..:: Globals ::..

const ArgumentBlockMap = Dict{ArgumentBlock, ArgumentBlock}
const PerturbationSet{T, N} = Dict{ConstantArgumentBlock{N}, T} where {
    T<:Types.RealArray}

# ..:: Data structures ::..

""" Conic clinear program main class. """
mutable struct ConicProgram <: AbstractConicProgram
    mdl::Model                # Core JuMP optimization model
    pars::Ref                 # Problem definition parameters
    x::VariableArgument       # Decision variable vector
    p::ConstantArgument       # Parameter vector
    cost::Ref{QuadraticCost}  # The cost function to minimize
    constraints::Constraints  # List of conic constraints

    _feasibility::Bool        # Flag if feasibility problem
    _solver::DataType         # The solver input to the constructor
    _solver_options::Types.Optional{Dict{String}} # Solver options

    """
        ConicProgram([pars][; solver, solver_options])

    Empty model constructor.

    # Arguments
    - `pars`: (optional )problem parameter structure. This can be anything, and
      it is passed down to the low-level functions defining the problem.

    # Keywords
    - `solver`: (optional) the numerical convex optimizer to use.
    - `solver_options`: (optional) options to pass to the numerical convex
      optimizer.

    # Returns
    - `prog`: the conic linear program data structure.
    """
    function ConicProgram(
        pars::Any=nothing;
        solver::DataType=ECOS.Optimizer,
        solver_options::Types.Optional{Dict{String}}=nothing)::ConicProgram

        # Configure JuMP model
        mdl = Model()
        set_optimizer(mdl, solver)
        if !isnothing(solver_options)
            for (key,val) in solver_options
                set_optimizer_attribute(mdl, key, val)
            end
        end

        # Variables and parameters
        pars = Ref(pars)
        x = VariableArgument()
        p = ConstantArgument()

        # Constraints
        constraints = Constraints()

        # Objective to minimize (empty for now)
        cost = Ref{QuadraticCost}()
        _feasibility = true

        # Combine everything into a conic program
        prog = new(mdl, pars, x, p, cost, constraints,
                   _feasibility, solver, solver_options)

        # Associate the arguments with the newly created program
        link!(x, prog)
        link!(p, prog)

        # Add a zero objective (feasibility problem)
        cost!(prog, feasibility_cost, [], [])

        return prog
    end # function
end # struct

# ..:: Methods ::..

"""
Get the underlying JuMP optimization model object, starting from various
structures composing the conic program.
"""
jump_model(prog::ConicProgram)::Model = prog.mdl

"""
    deconflict_name(new_name, existing_names)

Modify `new_name` so that it is not a duplicate of an existing name in the list
`existing_names`.

# Arguments
- `new_name`: the new name.
- `existing_names`: list of existing names.

# Returns
- `new_name`: updated `new_name` (only if necessary to deconflict).
"""
function deconflict_name(new_name::String,
                         existing_names::Vector{String})::String
    regex = Regex(@sprintf("^%s", new_name))
    regex_match = (s) -> !isempty(findall(regex, s))
    duplicate_count = length(findall(
        (this_name)->regex_match(this_name), existing_names))
    if duplicate_count>0
        new_name = @sprintf("%s%d", new_name, duplicate_count)
    end
    return new_name
end # function

"""
    push!(prog, kind, shape[; name])

Add a new argument block to the optimization program.

# Arguments
- `prog`: the optimization program.
- `kind`: the kind of argument (`VARIABLE` or `PARAMETER`).
- `shape...`: the shape of the argument block.

# Returns
- `block`: the new argument block.
"""
function Base.push!(prog::ConicProgram,
                    kind::ArgumentKind,
                    shape::Int...;
                    blk_name::Types.Optional{String}=nothing)::ArgumentBlock

    if !(kind in (VARIABLE, PARAMETER))
        err = SCPError(0, SCP_BAD_ARGUMENT,
                       "specify either VARIABLE or PARAMETER")
        throw(err)
    end

    z = (kind==VARIABLE) ? prog.x : prog.p

    # Assign the name
    if isnothing(blk_name)
        # Create a default name
        base_name = (kind==VARIABLE) ? "x" : "p"
        blk_name = base_name*@sprintf("%d", length(z)+1)
    else
        # Deconflict duplicate name by appending a number suffix
        all_names = [name(blk) for blk in z]
        blk_name = deconflict_name(blk_name, all_names)
    end

    block = push!(z, blk_name, shape...)

    return block
end

""" Specialize `push!` for variables. """
function variable!(prog::ConicProgram, shape::Int...;
                   name::Types.Optional{String}=nothing)::ArgumentBlock
    push!(prog, VARIABLE, shape...; blk_name=name)
end # function

""" Specialize `push!` for parameters. """
function parameter!(prog::ConicProgram, shape::Int...;
                    name::Types.Optional{String}=nothing)::ArgumentBlock
    push!(prog, PARAMETER, shape...; blk_name=name)
end # function

"""
    constraint!(prog, kind, f, x, p)

Create a conic constraint and add it to the problem. The heavy computation is
done by the user-supplied function `f`, which has to satisfy the requirements
of `DifferentiableFunction`.

# Arguments
- `prog`: the optimization program.
- `kind`: the cone type.
- `f`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks.
- `p`: the parameter argument blocks.

# Keywords
- `name`: (optional) a name for the constraint, which can be used to more
  easily search for it in the constraints list.
- `dual`: (optional) if true, then constrain f to lie inside the dual cone.

# Returns
- `new_constraint`: the newly added constraint.
"""
function constraint!(prog::ConicProgram,
                     kind::SuppotedCone,
                     f::Function,
                     x, p;
                     refname::Types.Optional{String}=nothing,
                     dual::Bool=false)::ConicConstraint
    x = VariableArgumentBlocks(collect(x))
    p = ConstantArgumentBlocks(collect(p))
    Axb = ProgramFunction(prog, x, p, f)

    # Assign the name
    if isnothing(refname)
        # Create a default name
        constraint_count = length(constraints(prog))
        refname = @sprintf("f%d", constraint_count+1)
    else
        # Deconflict duplicate name by appending a number suffix
        all_names = [name(C) for C in constraints(prog)]
        refname = deconflict_name(refname, all_names)
    end

    new_constraint = ConicConstraint(Axb, kind, prog; name=refname, dual=dual)
    push!(prog.constraints, new_constraint)
    return new_constraint
end # function

"""
    cost!(prog, J, x, p)

Set the cost of the conic program. The heavy computation is done by the
user-supplied function `J`, which has to satisfy the requirements of
`QuadraticDifferentiableFunction`.

# Arguments
- `prog`: the optimization program.
- `J`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks.
- `p`: the parameter argument blocks.

# Returns
- `new_cost`: the newly created cost.
"""
function cost!(prog::ConicProgram,
               J::Function,
               x, p)::QuadraticCost
    x = VariableArgumentBlocks(collect(x))
    p = ConstantArgumentBlocks(collect(p))
    J = ProgramFunction(prog, x, p, J)
    new_cost = QuadraticCost(J, prog)
    prog.cost[] = new_cost
    prog._feasibility = false
    return new_cost
end # function

"""
    new_argument(prog, shape, name, kind)

Add a new argument to the problem.

# Arguments
- `prog`: the conic program object.
- `shape`: the shape of the argument, as a tuple/vector/integer.
- `name`: the argument name.
- `kind`: either `VARIABLE` or `PARAMETER`.

# Returns
The newly created argument object.
"""
function new_argument(prog::ConicProgram,
                      shape,
                      name::Types.Optional{String},
                      kind::ArgumentKind)::ArgumentBlock
    f = (kind==VARIABLE) ? variable! : parameter!
    shape = collect(shape)
    return f(prog, shape...; name=name)
end # function

""" Check if this is a feasibility problem """
is_feasibility(prog::ConicProgram)::Bool = prog._feasibility

"""
    constraints(prg[, ref])

Get all or some of the constraints. If `ref` is a number or slice, then get
constraints from the list by regular array slicing operation. If `ref` is a
string, return all the constraints whose name contains the string `ref`.

# Arguments
- `prg`: the optimization problem.
- `ref`: (optional) which constraint to get.

# Returns
The constraint(s).
"""
function constraints(prg::ConicProgram,
                     ref=-1)::Union{Constraints, ConicConstraint}
    if typeof(ref)<:String
        # Search for all constraints the match `ref`
        match_list = Vector{ConicConstraint}(undef, 0)
        for constraint in prg.constraints
            if occursin(ref, name(constraint))
                push!(match_list, constraint)
            end
        end
        return match_list
    else
        # Get the constraint by numerical reference
        if ref>=0
            return prg.constraints[ref]
        else
            return prg.constraints
        end
    end
end # function

""" Get the optimization problem cost. """
cost(prg::ConicProgram)::QuadraticCost = prg.cost[]

"""
    solve!(prg)

Solve the optimization problem.

# Arguments
- `prg`: the optimization problem structure.

# Returns
- `status`: the termination status code.
"""
function solve!(prg::ConicProgram)::MOI.TerminationStatusCode
    mdl = jump_model(prg)
    optimize!(mdl)
    status = termination_status(mdl)
    return status
end # function

"""
    copy(blk, prg)

Copy the argument block (in aspects like shape, name, and scaling) to a
different optimization problem. The newly created argument gets inserted at the
end of the corresponding argument of the new problem

# Arguments
- `foo`: description.
- `prg`: the destination optimization problem for the new variable.

# Keywords
- `new_name`: (optional) a format string for creating the new name from the
  original block's name.
- `copyas`: (optional) force copy the block as a `VARIABLE` or `PARAMETER`.

# Returns
- `new_blk`: the newly created argument block.
"""
function Base.copy(blk::ArgumentBlock{T},
                   prg::ConicProgram;
                   new_name::String="%s",
                   copyas::Types.Optional{
                       ArgumentKind}=nothing)::ArgumentBlock where T

    blk_shape = size(blk)
    blk_kind = isnothing(copyas) ? kind(blk) : copyas
    blk_name = name(blk)
    new_name = @eval @sprintf($new_name, $blk_name)

    new_blk = new_argument(prg, blk_shape, new_name, blk_kind)
    apply_scaling!(new_blk, scale(blk))

    return new_blk
end # function

# ..:: Macros ::..

"""
    @new_variable(prog, shape, name)
    @new_variable(prog, shape)
    @new_variable(prog, name)
    @new_variable(prog)

The following macros specialize `@new_argument` for creating variables.
"""
macro new_variable(prog, shape, name)
    var = QuoteNode(VARIABLE)
    :( new_argument($(esc.([prog, shape, name, var])...)) )
end # macro

macro new_variable(prog, shape_or_name)
    var = QuoteNode(VARIABLE)
    if typeof(shape_or_name)<:String
        :( new_argument($(esc.([prog, 1, shape_or_name, var])...)) )
    else
        :( new_argument($(esc.([prog, shape_or_name, nothing, var])...)) )
    end
end # macro

macro new_variable(prog)
    :( new_argument($(esc(prog)), 1, nothing, VARIABLE) )
end # macro

"""
    @new_parameter(prog, shape, name)
    @new_parameter(prog, shape)
    @new_parameter(prog, name)
    @new_parameter(prog)

The following macros specialize `@new_argument` for creating parameters.
"""
macro new_parameter(prog, shape, name)
    var = QuoteNode(PARAMETER)
    :( new_argument($(esc.([prog, shape, name, var])...)) )
end # macro

macro new_parameter(prog, shape_or_name)
    var = QuoteNode(PARAMETER)
    if typeof(shape_or_name)<:String
        :( new_argument($(esc.([prog, 1, shape_or_name, var])...)) )
    else
        :( new_argument($(esc.([prog, shape_or_name, nothing, var])...)) )
    end
end # macro

macro new_parameter(prog)
    :( new_argument($(esc(prog)), 1, nothing, PARAMETER) )
end # macro

"""
    @add_constraint(prog, kind, name, f, x, p)
    @add_constraint(prog, kind, name, f, x)
    @add_constraint(prog, kind, f, x, p)
    @add_constraint(prog, kind, f, x)

Add a conic constraint to the optimization problem. This is just a wrapper of
the function `constraint!`, so look there for more info.

# Arguments
- `prog`: the optimization program.
- `kind`: the cone type.
- `name`: (optional) the constraint name.
- `f`: the core method that can compute the function value and its Jacobians.
- `x`: the variable argument blocks, as a vector/tuple/single element.
- `p`: (optional) the parameter argument blocks, as a vector/tuple/single
  element.

# Returns
The newly created `ConicConstraint` object.
"""
macro add_constraint(prog, kind, name, f, x, p)
    :( constraint!($(esc.([prog, kind, f, x, p])...);
                   refname=$name) )
end # macro

macro add_constraint(prog, kind, name_f, f_x, x_p)
    if typeof(name_f)<:String
        :( constraint!($(esc.([prog, kind, f_x, x_p, []])...);
                       refname=$name_f) )
    else
        :( constraint!($(esc.([prog, kind, name_f, f_x, x_p])...)) )
    end
end # macro

macro add_constraint(prog, kind, f, x)
    :( constraint!($(esc.([prog, kind, f, x, []])...)) )
end # macro

"""
    @add_dual_constraint(prog, kind, name, f, x, p)
    @add_dual_constraint(prog, kind, name, f, x)
    @add_dual_constraint(prog, kind, f, x, p)
    @add_dual_constraint(prog, kind, f, x)

These macros work exactly like `@add_constraint`, except the value `f` is
imposed to lie inside the dual of the cone `kind`.
"""
macro add_dual_constraint(prog, kind, name, f, x, p)
    :( constraint!($(esc.([prog, kind, f, x, p])...);
                   refname=$name, dual=true) )
end # macro

macro add_dual_constraint(prog, kind, name_f, f_x, x_p)
    if typeof(name_f)<:String
        :( constraint!($(esc.([prog, kind, f_x, x_p, []])...);
                       refname=$name_f, dual=true) )
    else
        :( constraint!($(esc.([prog, kind, name_f, f_x, x_p])...)) )
    end
end # macro

macro add_dual_constraint(prog, kind, f, x)
    :( constraint!($(esc.([prog, kind, f, x, []])...);
                   dual=true) )
end # macro

"""
    @set_cost(prog, J, x, p)
    @set_cost(prog, J, x)
    @set_feasibility(prog, J)

Set the optimization problem cost. This is just a wrapper of the function
`cost!`, so look there for more info. When both `x` and `p` arguments are
ommitted, the cost is constant so this must be a feasibility problem. In this
case, the macro name is `@set_feasibility`

# Arguments
- `prog`: the optimization program.
- `J`: the core method that can compute the function value and its Jacobians.
- `x`: (optional) the variable argument blocks, as a vector/tuple/single
  element.
- `p`: (optional) the parameter argument blocks, as a vector/tuple/single
  element.

# Returns
- `bar`: description.
"""
macro set_cost(prog, J, x, p)
    :( cost!($(esc.([prog, J, x, p])...)) )
end # macro

macro set_cost(prog, J, x)
    :( cost!($(esc.([prog, J, x, []])...)) )
end # macro

macro set_feasibility(prog)
    quote
        $(esc(prog))._feasibility = true
        cost!($(esc(prog)), feasibility_cost, [], [])
    end
end # macro
