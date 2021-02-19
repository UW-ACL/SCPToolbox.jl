# Variable types used in the code

using JuMP
using Printf

# Possible SCvx-specific solution statuses
@enum(SCvxStatus,
      SCVX_SOLVED,
      SCVX_FAILED,
      SCVX_EMPTY_VARIABLE,
      SCVX_SCALING_FAILED,
      SCVX_GUESS_PROJECTION_FAILED)

const T_Bool = Bool
const T_Int = Int
const T_String = String
const T_Real = Float64
const T_Symbol = Symbol

const T_IntVector = Vector{T_Int}
const T_IntRange = UnitRange{T_Int}

__types_f(n) = Array{T_Real, n}
const T_RealVector = __types_f(1)
const T_RealMatrix = __types_f(2)
const T_RealTensor = __types_f(3)

const T_OptiModel = Model

__types_f(n) = Union{
    Array{T_Real, n},
    Array{VariableRef, n},
    Array{GenericAffExpr{T_Real, VariableRef}, n}}
const T_OptiVarVector = __types_f(1)
const T_OptiVarMatrix = __types_f(2)

__types_f(n) = Array{ConstraintRef, n}
const T_ConstraintVector = __types_f(1)
const T_ConstraintMatrix = __types_f(2)

const T_Objective = Union{Missing,
                          Float64,
                          VariableRef,
                          GenericAffExpr{T_Real, VariableRef},
                          GenericQuadExpr{T_Real, VariableRef}}

const T_ExitStatus = Union{SCvxStatus, MOI.TerminationStatusCode}

const T_Function = Union{Nothing, Function}
const T_FunctionVector = Vector{T_Function}

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
function Table(
    def::Vector{Tuple{T_Symbol, T_String, T_String, T_Int}},
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
