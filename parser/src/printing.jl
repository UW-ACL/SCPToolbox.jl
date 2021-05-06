#= Pretty printing of parser data structures.

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

using JuMP

"""
    print_indices(id[, limit])

Print an vector of array linear indices, with a limit on how many to print in
total.

# Arguments
- `id`: the index vector.
- `limit`: (optional) half the maximum number of indices to show (at start and
  end).

# Returns
- `ids`: the index vector in string format, ready to print.
"""
function print_indices(id::LocationIndices, limit::Int=3)::String

    # Check if a single number
    if ndims(id)==0 || length(id)==1
        ids = @sprintf("%d", id[1])
        return ids
    end

    # Check if contiguous range
    iscontig = !any(diff(id).!=1)
    if iscontig
        ids = @sprintf("%d:%d", id[1], id[end])
        return ids
    end

    # Print a limited number of indices
    printarr = (arr) -> join(map((x)->@sprintf("%s", x), arr), ",")
    if length(id)>2*limit
        v0 = printarr(id[1:limit])
        vf = printarr(id[end-limit+1:end])
        ids = @sprintf("%s,...,%s", v0, vf)
    else
        ids = printarr(id)
    end
    return ids
end # function

"""
    _string_round(f)

Override JuMP's low level floating point printing command to print a smller
number of decimal places.

# Arguments
- `f`: the floating point number.

# Returns
The rounded floating point number as a string.
"""
function JuMP._string_round(f::Float64)
    iszero(f) && return "0" # strip sign off zero
    precision = 3 # Number of decimal points
    # str = string(round(f; digits=precision))
    fmt = @sprintf("%%.%de", precision)
    str = @eval @sprintf($fmt, $f)
    return length(str) >= 2 && str[end-1:end] == ".0" ? str[1:end-2] : str
end

"""
    print_array(io, z)

Print the value of an abstract array, which can be a variable vector.

# Arguments
- `io`: stream object.
- `z`: the vector to be printed.
"""
function print_array(io::IO, z::AbstractArray)::Nothing
    compact = get(io, :compact, false) #noinfo
    io_value = IOBuffer()
    max_rows, max_cols = 10, 50
    indent = " "^get(io, :indent, 0)
    io2_value = IOContext(io_value,
                          :limit=>true,
                          :displaysize=>(max_rows, max_cols),
                          :compact=>true)
    show(io2_value, MIME("text/plain"), z)
    value_str = String(take!(io_value))
    # Print line by line
    rows = split(value_str, "\n")[2:end]
    for i=1:length(rows)
        row = rows[i]
        if length(row) > max_cols
            row = row[1:max_cols]*" …"
        end
        newline = (i==length(rows)) ? "" : "\n"
        @printf(io, "%s%s%s", indent, row, newline)
    end
    return nothing
end

"""
    show(io, cone[; z])

Pretty print the cone.

# Arguments
- `io`: stream object.
- `cone`: the cone constraint to be printed.

# Keywords
- `z`: (optional) the string to use for the value constrained inside the cone.
"""
function Base.show(io::IO, cone::ConvexCone; z::String="z")::Nothing
    compact = get(io, :compact, false) #noinfo

    if kind(cone)==:free
        @printf("Unconstrained %s\n", z)
    else
        cone_description = Dict(
            :zero => "{z : z=0}",
            :nonpos => "{z : z≤0}",
            :l1 => "{(t, x)∈ℝ×ℝⁿ : ‖x‖₁≤t}",
            :soc => "{(t, x)∈ℝ×ℝⁿ : ‖x‖₂≤t}",
            :linf => "{(t, x)∈ℝ×ℝⁿ : ‖x‖∞≤t}",
            :geom => "{(t, x)∈ℝ×ℝⁿ : (x₁x₂⋯xₙ)^{1/n}≥t}",
            :exp => "{(x,y,w)∈ℝ³ : y⋅e^{x/y}≤w, y>0}")

        @printf("Cone %s∈K, where:\n", z)
        @printf("K is a %s cone, %s\n", CONE_NAMES[cone.kind],
                cone_description[cone.kind])
    end

    if !compact
        # Print the value of z
        @printf("%s = \n", z)
        io2 = IOContext(io, :indent=>1)
        print_array(io2, cone.z)
    end

    return nothing
end # function

"""
Get string description of the kind of function this is.
"""
function function_kind(F::ProgramFunction)::String
    value_type = typeof(F.f.out[].value)
    quadratic_type = Types.QExpr
    kind = (value_type<:quadratic_type) ? "Quadratic" : "Affine"
    return kind
end # function

"""
    show(io, F)

Pretty print an affine function.

# Arguments
- `io`: stream object.
- `F`: the affine function.
"""
function Base.show(io::IO, F::ProgramFunction)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf(io, "%s function\n", function_kind(F))

    all_args = vcat(F.x, F.p)
    if isempty(all_args)
        @printf("Constant\n")
    else
        @printf("Arguments:\n")
        longest_name = maximum([length(arg.name) for arg in all_args])
        name_fmt = @sprintf("  %%-%ds", longest_name)
        for arg in all_args
            @eval @printf($(name_fmt), $(arg).name)
            @printf(" (block %d) : ", arg.blid)
            @printf("%s\n", print_indices(arg.elid))
        end
    end

    if !compact
        try
            f_value = value(F)
            @printf("Current value =\n")
            io2 = IOContext(io, :indent=>1)
            print_array(io2, f_value)
        catch
            @printf("Not evaluated yet")
        end
    end

    return nothing
end # function

"""
    show(io, F)

Pretty print the cost function.

# Arguments
- `io`: stream object.
- `cost`: the cost function.
"""
function Base.show(io::IO, cost::QuadraticCost)::Nothing
    compact = get(io, :compact, false) #noinfo
    show(io, cost.J)
    return nothing
end # function

"""
    show(io, f)

Pretty print a differentiable function object.

# Arguments
- `io`: stream object.
- `f`: the differentiable function object.
"""
function Base.show(io::IO, f::DifferentiableFunction)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf("Differentiable function:\n")
    @printf("  %d variable arguments\n", f.xargs)
    @printf("  %d constant arguments\n", f.pargs)
    @printf("  Parameter container: %s\n", typeof(f.consts[]))

    if f.evaluated
        @printf("  %d Jacobians available\n", length(all_jacobians(f.out[])))
        @printf("  Current value =\n")
        io2 = IOContext(io, :indent=>3)
        print_array(io2, value(f))
    else
        @printf("  Not evaluated.")
    end

    return nothing
end # function

"""
    show(io, cone)

Pretty print a conic constraint.

# Arguments
- `io`: stream object.
- `cone`: the affine function.
"""
function Base.show(io::IO, cone::ConicConstraint)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf(io, "Name: %s\n", name(cone))

    show(io, cone.K; z="f(x,p)")
    @printf(io, "\n")

    io2 = IOContext(io, :compact=>true)
    show(io2, cone.f)

    return nothing
end # function

"""
    show(io, cone)

Pretty print a conic constraint.

# Arguments
- `io`: stream object.
- `cone`: the affine function.
"""
function Base.show(io::IO, constraints::Constraints)::Nothing
    compact = get(io, :compact, false) #noinfo
    indent = " "^get(io, :indent, 0)

    @printf(io, "%s%d constraints", indent, length(constraints))

    # Print a more detailed list of constraints
    # Collect cone types
    cone_count = Dict(cone=>0 for cone in SUPPORTED_CONES)
    for constraint in constraints
        cone_count[kind(constraint)] += 1
    end
    # Print out
    for cone in SUPPORTED_CONES #noinfo
        count = cone_count[cone] #noinfo
        name = CONE_NAMES[cone] #noinfo
        if count>0
            @printf(io, "\n%s  %d %s cones", indent, count, name)
        end
    end

    if !compact
        @printf("\n\n")
        for cone in constraints #noinfo
            show(io, cone)
            @printf(io, "\n")
        end
    end

    return nothing
end # function

"""
    show(io, sc)

Pretty print the argument block scaling.

# Arguments
- `io`: stream object.
- `sc`: the scaling object.
"""
function Base.show(io::IO, sc::Scaling)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf(io, "Affine scaling x=(S.*xh).+c\n")

    @printf(io, "S =\n")
    io2 = IOContext(io, :indent=>1, :compact=>true)
    print_array(io2, dilation(sc))
    @printf(io, "\n")

    @printf(io, "c =\n")
    io2 = IOContext(io, :indent=>1, :compact=>true)
    print_array(io2, offset(sc))

    return nothing
end # function


"""
    show(io, arg)

Pretty print an argument block.

# Arguments
- `io`: stream object.
- `blk`: the argument block.
"""
function Base.show(io::IO, blk::ArgumentBlock{T})::Nothing where T
    compact = get(io, :compact, false) #noinfo

    isvar = T==AtomicVariable
    dim = ndims(blk)
    qualifier = Dict(0=>"Scalar", 1=>"Vector", 2=>"Matrix", 3=>"Tensor")
    if dim<=3
        qualifier = qualifier[dim]
    else
        qualifier = "N-dimensional"
    end

    kind = isvar ? "variable" : "parameter"

    @printf(io, "%s %s\n", qualifier, kind)
    @printf(io, "  %d elements\n", length(blk))
    @printf(io, "  %s shape\n", size(blk))
    @printf(io, "  Name: %s\n", blk.name)
    @printf(io, "  Block index in stack: %d\n", blk.blid)
    @printf(io, "  Indices in stack: %s\n", print_indices(blk.elid))
    @printf(io, "  Type: %s\n", typeof(blk.value))
    @printf(io, "  Value =\n")
    io2 = IOContext(io, :indent=>4, :compact=>true)
    print_array(io2, blk.value)

    return nothing
end # function

"""
    show(io, arg)

Pretty print an argument.

# Arguments
- `io`: stream object.
- `arg`: argument structure.
"""
function Base.show(io::IO, arg::Argument{T})::Nothing where {T<:AtomicArgument}
    compact = get(io, :compact, false) #noinfo

    isvar = T<:AtomicVariable
    kind = isvar ? "Variable" : "Parameter"
    n_blocks = length(arg)
    indent = " "^get(io, :indent, 0)

    @printf(io, "%s%s argument\n", indent, kind)
    @printf(io, "%s  %d elements\n", indent, numel(arg))
    @printf(io, "%s  %d blocks\n", indent, n_blocks)

    if n_blocks==0
        return nothing
    end

    ids = (i) -> arg.blocks[i].elid
    make_span_str = (i) -> print_indices(ids(i))
    span_str = [make_span_str(i) for i=1:n_blocks]
    max_span_sz = maximum(length.(span_str))
    for i = 1:n_blocks
        newline = (i==n_blocks) ? "" : "\n"
        span_str = make_span_str(i)
        span_diff = max_span_sz-length(span_str)
        span_str = span_str*(" "^span_diff)
        @printf(io, "%s   %d) %s ... %s%s",
                indent, i, span_str, arg.blocks[i].name, newline)
    end

    return nothing
end # function

"""
    show(io, prog)

Pretty print the conic program.

# Arguments
- `io`: stream object.
- `prog`: the conic program data structure.
"""
function Base.show(io::IO, prog::ConicProgram)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf(io, "Conic linear program\n\n")
    if is_feasibility(prog)
        @printf("  Feasibility problem\n")
    else
        kind = function_kind(cost(prog).J)
        @printf("  %s cost function\n", kind)
    end
    @printf(io, "  %d variables (%d blocks)\n", length(prog.x),
            length(prog.x))
    @printf(io, "  %d parameters (%d blocks)\n", length(prog.p),
            length(prog.p))

    io2 = IOContext(io, :indent=>2, :compact=>true)
    show(io2, constraints(prog))

    # Print a more detailed list of variables and parameters
    if !compact
        io2 = IOContext(io, :indent=>2)
        @printf("\n\n")
        show(io2, prog.x)
        @printf("\n\n")
        show(io2, prog.p)
    end

    return nothing
end # function

"""
    show(io, mime, obj)

A wrapper of the above `show` functions when `MIME` argument is passed in (it
is just ignored!).

# Arguments
- `io`: stream object.
- `obj`: the object to be printed.
- `args...`: any other arguments the underlying `show` function accepts.

# Keywords
- `kwargs...`: any other keyword arguments the underlying `show` function
  accepts.
"""
Base.show(io::IO, ::MIME"text/plain",
          constraints::Union{
              ConvexCone,
              ProgramFunction,
              DifferentiableFunction,
              ConicConstraint,
              Constraints,
              Scaling,
              ArgumentBlock,
              Argument,
              ConicProgram},
          args...; kwargs...) = show(io, constraints, args...; kwargs...)
