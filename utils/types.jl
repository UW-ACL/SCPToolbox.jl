# Variable types used in the code

using LinearAlgebra
using JuMP

# Possible SCvx-specific solution statuses
@enum(SCvxStatus,
      SCVX_SOLVED,
      SCVX_FAILED,
      SCVX_EMPTY_VARIABLE,
      SCVX_SCALING_FAILED,
      SCVX_GUESS_PROJECTION_FAILED)

# ..:: Basic types ::..

const T_Bool = Bool
const T_Int = Int
const T_String = String
const T_Real = Float64
const T_Symbol = Symbol

const T_IntVector = Vector{T_Int}
const T_IntRange = UnitRange{T_Int}

const T_RealArray = Array{T_Real}
__types_f(n) = T_RealArray{n}
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

# ..:: Composite types ::..

#= Quaternion using Hamiltonian convention.

In vectorized form/indexing, use the scalar last convention. =#
struct T_Quaternion
    v::T_RealVector # Vector part
    w::T_Real       # Scalar part

    #= Basic constructor.

    Args:
        w: the scalar part.
        v: the vector part.

    Returns:
        q: the quaternion. =#
    function T_Quaternion(v::T_RealVector, w::T_Real)::T_Quaternion
        if length(v)!=3
            err = ArgumentError("ERROR: quaternion is a 4-element object.")
            throw(err)
        end

        q = new(v, w)

        return q
    end

    #= (Pure) quaternion constructor from a vector.

    Args:
        v: the vector part or the full quaternion in vector form.

    Returns:
        q: the pure quaternion. =#
    function T_Quaternion(v::T_RealVector)::T_Quaternion
        if length(v)!=3 && length(v)!=4
            msg = string("ERROR: cannot construct a quaternion from ",
                         "fewer than 3 or more than 4 elements.")
            err = ArgumentError(msg)
            throw(err)
        end

        if length(v)==3
            q = new(v, 0.0)
        else
            q = new(v[1:3], v[4])
        end

        return q
    end

    #= Unit quaternion from an angle-axis attitude parametrization.

    Args:
        α: the angle (in radians).
        a: the axis (internally normalized to a unit norm).

    Returns:
        q: the unit quaternion. =#
    function T_Quaternion(α::T_Real, a::T_RealVector)::T_Quaternion
        if length(a)!=3
            msg = string("ERROR: axis must be in R^3.")
            err = ArgumentError(msg)
            throw(err)
        end

        a /= norm(a)
        v = a*sin(α/2)
        w = cos(α/2)
        q = new(v, w)

        return q
    end
end

# ..:: Data structures ::..

#= Continuous-time trajectory data structure. =#
struct ContinuousTimeTrajectory
    t::T_RealVector  # The trajectory time nodes
    x::T_RealArray   # The trajectory values at the corresponding times
    # Interpolation type between time nodes, possible values are:
    #   :linear (linear interpolation)
    interp::T_Symbol

    #= Constructor.

    Args:
        t: the trajectory time nodes.
        x: the trajectory values at the corresponding times.
        interp: the interpolation method.

    Returns:
        traj: the continuous-time trajectory. =#
    function ContinuousTimeTrajectory(
        t::T_RealVector,
        x::T_RealArray,
        interp::T_Symbol)

        if !(interp in [:linear])
            err = ArgumentError("ERROR: unknown trajectory interpolation type.")
            throw(err)
        end

        traj = new(t, x, interp)

        return traj
    end
end

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
