# Variable types used in the code

using JuMP
using Printf

abstract type AbstractTrajectoryProblem end

# Possible SCvx-specific solution statuses
@enum(SCvxStatus,
      PENALTY_CHECK_FAILED)

T_Bool = Bool
T_Int = Int
T_String = String
T_Real = Float64
T_Symbol = Symbol


T_RealOrNothing = Union{T_Real, Nothing}
T_IntVector = Vector{T_Int}
T_RealArrayLike = Array{T_Real}
T_RealVector = T_RealArrayLike{1}
T_RealMatrix = T_RealArrayLike{2}
T_RealTensor = T_RealArrayLike{3}
T_IntRange = UnitRange{T_Int}
T_OptiModel = Model
T_OptiVar = VariableRef
T_OptiVarAffTransf = GenericAffExpr{T_Real,VariableRef}
T_OptiVarVector = Vector{T_OptiVar}
T_OptiVarMatrix = Matrix{T_OptiVar}
T_OptiVarAffTransfVector = Vector{T_OptiVarAffTransf}
T_OptiVarAffTransfMatrix = Matrix{T_OptiVarAffTransf}
T_RealOrOptiVarVector = Union{T_RealVector, T_OptiVarAffTransfVector}
T_RealOrOptiVarMatrix = Union{T_RealMatrix, T_OptiVarAffTransfMatrix}
T_Constraint = Union{ConstraintRef, Nothing}
T_ConstraintVector = Vector{T_Constraint}
T_ConstraintMatrix = Matrix{T_Constraint}
T_Objective = Union{Nothing,
                    Float64,
                    T_OptiVar,
                    GenericAffExpr{T_Real,T_OptiVar},
                    GenericQuadExpr{T_Real,T_OptiVar}}
T_ExitStatus = Union{SCvxStatus, MOI.TerminationStatusCode}

#= Iteration progress information table to be printed in REPL. =#
mutable struct Table
    headings::Array{T_String}      # Column headings
    sorting::Dict{T_Symbol, T_Int} # Column order
    fmt::Dict{T_Symbol, T_String}  # Column value format
    row::T_String                  # The full row format
    # >> Private members <<
    __head_print::T_Bool # Flag whether the column headings have been printed
    __colw::T_IntVector  # Column widths
end

#= Error exception in SCvx algorithm. Hopefully it doesn't happen! =#
struct SCvxError <: Exception
    k::T_Int           # At what discrete time step the error occured
    status::SCvxStatus # Error status code
    msg::T_String      # Error message
end

# ..:: Constructors ::..

#= Table constructor.

Args:
    def: an array of column definitions, where each element is a tuple of:
        [1]: unique symbol referencing this colum.
        [2]: column heading.
        [3]: column format (for sprintf to convert into a string).
        [4]: column width.

Returns:
    table: the table structure. =#
function Table(def::Vector{Tuple{T_Symbol,
                                 T_String,
                                 T_String,
                                 T_Int}},
               separator::T_String="|")::Table

    # Initialize
    headings = Array{T_String}(undef, 0)
    sorting = Dict{T_Symbol, T_Int}()
    fmt = Dict{T_Symbol, T_String}()
    row = ""
    colw = T_IntVector(undef, 0)
    args = (headings, sorting, fmt, colw, separator)

    # Add each column
    for d in def
        sym, head, n2s, w = d[1], d[2], d[3], d[4]
        row = _types__add_table_column!(sym, head, n2s, w, args..., row)
    end

    # Default values
    head_print = true

    table = Table(headings, sorting, fmt, row, head_print, colw)

    return table
end

# ..:: Public methods ::..

#= Print row of table.

Args:
    row: table row specification.
    table: the table specification. =#
function print(row::Dict{T_Symbol, T}, table::Table)::Nothing where {T}
    # Assign values to table columns
    values = fill("", length(table.headings))
    for (k, v) in row
        val_fmt = table.fmt[k]
        values[table.sorting[k]] = @eval @sprintf($val_fmt, $v)
    end

    if table.__head_print==true
        table.__head_print = false
        # Print the columnd headers
        top = @eval @printf($(table.row), $(table.headings)...)
        println()

        _types__table_print_hrule(table)
    end

    msg = @eval @printf($(table.row), $values...)
    println()

    return nothing
end

# ..:: Private methods ::..

#= Add a new column to the iteration info table.

Args:
    col_sym: symbol that references this column.
    col_heading: the column heading.
    col_fmt: the column format (for sprintf to convert from native to string).
    col_width: the column's width.
    headings: the existing array of table column headings.
    sorting: the column sort order, so that we can re-define column ordering
        seamlessly.
    colw: vector of column widths.
    fmt: the existing string of table column format.
    sep: table column separator.
    row: the complete row format specification to add this new column to.

Returns:
    fmt: the updated string of table column format. =#
function _types__add_table_column!(
    col_sym::Symbol,
    col_heading::T_String,
    col_fmt::T_String,
    col_width::T_Int,
    headings::Vector{T_String},
    sorting::Dict{Symbol,T_Int},
    fmt::Dict{Symbol,T_String},
    colw::T_IntVector,
    sep::T_String,
    row::T_String)::T_String

    # Column separator
    separator = (length(row)==0) ? "" : string(" ", sep, " ")

    push!(headings, col_heading)
    push!(sorting, col_sym => length(headings))
    push!(fmt, col_sym => col_fmt)
    push!(colw, col_width)

    row = string(row, separator, "%-", col_width, "s")

    return row
end

#= Print table row horizontal separator line.

Args:
    table: the table specification. =#
function _types__table_print_hrule(table::Table)::Nothing
    hrule = ""
    num_cols = length(table.__colw)
    for i = 1:num_cols
        hrule = string(hrule, repeat("-", table.__colw[i]))
        if i < num_cols
            hrule = string(hrule, "-+-")
        end
    end
    println(hrule)

    return nothing
end
