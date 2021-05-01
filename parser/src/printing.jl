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
    show(z)

Print the value of an abstract array, which can be a variable vector.

# Arguments
- `z`: the vector to be printed.
"""
function Base.show(io::IO, z::AbstractArray)::Nothing
    compact = get(io, :compact, false) #noinfo
    io_value = IOBuffer()
    indent = " "^get(io, :indent, 0)
    io2_value = IOContext(io_value, :limit=>true, :displaysize=>(10, 50))
    show(io2_value, MIME("text/plain"), z)
    value_str = String(take!(io_value))
    # Print line by line
    rows = split(value_str, "\n")[2:end]
    for i=1:length(rows)
        row = rows[i]
        newline = (i==length(rows)) ? "" : "\n"
        @printf(io, "%s%s%s", indent, row, newline)
    end
    return nothing
end

"""
    show(cone)

Pretty print the cone.

# Arguments
- `cone`: the cone constraint to be printed.

# Keywords
- `z`: (optional) the string to use for the value constrained inside the cone.
"""
function Base.show(io::IO, cone::ConvexCone;
                   z::String="z")::Nothing
    compact = get(io, :compact, false) #noinfo

    cone_description = Dict(
        :zero => ("zero", "{z : z=0}"),
        :nonpos => ("nonpositive orthant", "{z : z≤0}"),
        :l1 => ("one-norm", "{(t, x)∈ℝ×ℝⁿ : ‖x‖₁≤t}"),
        :soc => ("second-order", "{(t, x)∈ℝ×ℝⁿ : ‖x‖₂≤t}"),
        :linf => ("inf-norm", "{(t, x)∈ℝ×ℝⁿ : ‖x‖∞≤t}"),
        :geom => ("geometric", "{(t, x)∈ℝ×ℝⁿ : (x₁x₂⋯xₙ)^{1/n}≥t}"),
        :exp => ("exponential", "{(x,y,w)∈ℝ³ : y⋅e^{x/y}≤w, y>0}"))

    @printf("Cone %s∈K, where:\n", z)
    @printf("K is a %s cone, %s\n", cone_description[cone.kind]...)

    if !compact
        # Print the value of z
        @printf("%s = \n", z)
        io2 = IOContext(io, :indent=>3)
        show(io2, cone.z)
    end

    return nothing
end # function

"""
    show(io, F)

Pretty print an affine function.

# Arguments
- `F`: the affine function.
"""
function Base.show(io::IO, F::AffineFunction)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf("Arguments of f:\n")
    all_args = vcat(F.x, F.p)
    longest_name = maximum([length(arg.name) for arg in all_args])
    name_fmt = @sprintf("  %%-%ds", longest_name)
    for arg in all_args
        @eval @printf($(name_fmt), $(arg).name)
        @printf(" (block %d) : ", arg.blid)
        @printf("%s\n", print_indices(arg.elid))
    end

    if !compact
        try
            f_value = value(F)
            @printf("Current value =\n")
            io2 = IOContext(io, :indent=>4)
            show(io2, f_value)
        catch
            @printf("Not evaluated yet")
        end
    end

    return nothing
end # function

"""
    show(io, cone)

Pretty print a conic constraint.

# Arguments
- `cone`: the affine function.
"""
function Base.show(io::IO, cone::ConicConstraint)::Nothing
    compact = get(io, :compact, false) #noinfo

    show(io, cone.K; z="f(x,p)")
    println()

    io2 = IOContext(io, :compact=>true)
    show(io2, cone.f)

    return nothing
end # function

"""
    show(io, arg)

Pretty print a conic constraint.

# Arguments
- `arg`: the argument.
"""
function Base.show(io::IO, arg::ArgumentBlock{T})::Nothing where T
    compact = get(io, :compact, false) #noinfo

    isvar = T==AtomicVariable
    dim = ndims(arg)
    qualifier = Dict(0=>"Scalar", 1=>"Vector", 2=>"Matrix", 3=>"Tensor")
    if dim<=3
        qualifier = qualifier[dim]
    else
        qualifier = "N-dimensional"
    end

    kind = isvar ? "variable" : "parameter"

    @printf(io, "%s %s\n", qualifier, kind)
    @printf(io, "  %d elements\n", length(arg))
    @printf(io, "  %s shape\n", size(arg))
    @printf(io, "  Name: %s\n", arg.name)
    @printf(io, "  Block index in stack: %d\n", arg.blid)
    @printf(io, "  Indices in stack: %s\n", print_indices(arg.elid))
    @printf(io, "  Type: %s\n", typeof(arg.value))
    @printf(io, "  Value =\n")
    io2 = IOContext(io, :indent=>4, :compact=>true)
    show(io2, arg.value)

    return nothing
end # function

Base.display(arg::ArgumentBlock) = show(stdout, arg)

"""
    show(io, arg)

Pretty print an argument.

# Arguments
- `arg`: argument structure.
"""
function Base.show(io::IO, arg::Argument{T})::Nothing where {T<:AtomicArgument}
    compact = get(io, :compact, false) #noinfo

    isvar = T<:AtomicVariable
    kind = isvar ? "Variable" : "Parameter"
    n_blocks = blocks(arg)
    indent = " "^get(io, :indent, 0)

    @printf(io, "%s%s argument\n", indent, kind)
    @printf(io, "%s  %d elements\n", indent, length(arg))
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
- `prog`: the conic program data structure.
"""
function Base.show(io::IO, prog::ConicProgram)::Nothing
    compact = get(io, :compact, false) #noinfo

    @printf(io, "Conic linear program\n")
    @printf(io, "  %d variables (%d blocks)\n", length(prog.x), blocks(prog.x))
    @printf(io, "  %d parameters (%d blocks)", length(prog.p), blocks(prog.p))

    if !compact
        io2 = IOContext(io, :indent=>2)
        @printf("\n\n")
        show(io2, prog.x)
        @printf("\n\n")
        show(io2, prog.p)
    end

    return nothing
end # function
