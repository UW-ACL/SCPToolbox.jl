# Variable types used in the code

using JuMP

abstract type AbstractTrajectoryProblem end

T_Bool = Bool
T_Int = Int
T_String = String
T_Real = Float64
T_Symbol = Symbol
T_RealOrNothing = Union{T_Real, Nothing}
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
T_Constraint = Union{ConstraintRef, Nothing}
T_ConstraintVector = Vector{T_Constraint}
T_ConstraintMatrix = Matrix{T_Constraint}
T_Objective = Union{Nothing,
                    Float64,
                    T_OptiVar,
                    GenericAffExpr{T_Real,T_OptiVar},
                    GenericQuadExpr{T_Real,T_OptiVar}}

struct Table
    headings::Array{T_String}      # Column headings
    sorting::Dict{T_Symbol, T_Int} # Column order
    fmt::Dict{T_Symbol, T_String}  # Column value format
    row::T_String                  # The full row format
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
    args = (headings, sorting, fmt, separator)

    # Add each column
    for d in def
        sym, head, n2s, w = d[1], d[2], d[3], d[4]
        row = _types__add_table_column!(sym, head, n2s, w, args..., row)
    end

    table = Table(headings, sorting, fmt, row)

    return table
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
    fmt: the existing string of table column format.

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
    sep::T_String,
    row::T_String)::T_String

    # Column separator
    separator = (length(row)==0) ? "" : string(sep, " ")

    push!(headings, col_heading)
    push!(sorting, col_sym => length(headings))
    push!(fmt, col_sym => col_fmt)

    row = string(row, separator, "%-", col_width, "s")

    return row
end
