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

const T_BoolVector = Vector{T_Bool}
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

const T_ElementIndex = Union{T_Int, T_IntRange, Colon}
const T_SpecialIntegrationActions = Vector{Tuple{T_ElementIndex, T_Function}}
const T_SIA = T_SpecialIntegrationActions # Alias

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
            q = T_Quaternion(v, 0.0)
        else
            q = T_Quaternion(v[1:3], v[4])
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
        q = T_Quaternion(v, w)

        return q
    end
end

#= Ellipsoid geometric object.

Ellipsoid set = {x : norm(H*(x-c), 2) <= 1},

where H is a positive definite matrix which defines the ellipsoid shape,
and c is the ellipsoid's center. =#
struct T_Ellipsoid
    H::T_RealMatrix # Ellipsoid shape matrix
    c::T_RealVector # Ellipsoid center

    #= Basic constructor.

    Args:
        H: ellipsoid shape matrix.
        c: ellipsoid center

    Returns:
        E: the ellipsoid. =#
    function T_Ellipsoid(H::T_RealMatrix,
                         c::T_RealVector)::T_Ellipsoid
        if size(H,2)!=length(c)
            err = ArgumentError("ERROR: matrix size mismatch.")
            throw(err)
        end
        E = new(H, c)
        return E
    end
end

#= Hyperrectangle geometric object.

Hyperrectangle set H = {x : l <= x <= u} =#
struct T_Hyperrectangle
    n::T_Int        # Ambient space dimension
    l::T_RealVector # Lower bound ("lower-left" vertex)
    u::T_RealVector # Upper bound ("upper-right" vertex)
    # >> Scaling x=s.*y+c such that H maps to {y: -1 <= y <= 1} <<
    s::T_RealVector # Dilation
    c::T_RealVector # Offset

    #= Basic constructor.

    Args:
        l: the lower-left vertex.
        u: the upper-right vertex.

    Returns:
        H: the hyperrectangle set. =#
    function T_Hyperrectangle(l::T_RealVector,
                              u::T_RealVector)::T_Hyperrectangle
        if length(l)!=length(u)
            err = ArgumentError("ERROR: vertex dimension mismatch.")
            throw(err)
        end
        n = length(l)
        s = (u-l)/2
        c = (u+l)/2
        H = new(n, l, u, s, c)
        return H
    end

    #= Constructor from axis ranges.

    Args:
        range: the (min, max) range of values in the set along
          axis 1, 2, etc.

    Returns:
        H: the hyperrectangle set. =#
    function T_Hyperrectangle(
        range::Tuple{T_Real, T_Real}...)::T_Hyperrectangle

        n = length(range)
        l = T_RealVector([range[i][1] for i=1:n])
        u = T_RealVector([range[i][2] for i=1:n])

        H = T_Hyperrectangle(l, u)

        return H
    end

    #= Extrusion-like constructor for a 3D rectangular prism.

    Think about it like extruding a 3D rectangular prism in a Computer Aided
    Design (CAD) software. You create a 2D rectangle centered at c with
    dimensions width x height. You then extrude "forward" along the +z axis by
    the depth value. Afterwards, you yaw the rectangle by ±90 degrees in yaw,
    pitch, and roll.

    Args:
        offset: the rectangular base center (aka centroid).
        width: rectangular base width (along x).
        height: rectangular base height (along y).
        depth: extrusion depth (along +z).
        yaw: (optional) the Tait-Bryan yaw angle (in degrees).
        pitch: (optional) the Tait-Bryan pitch angle (in degrees).
        roll: (optional) the Tait-Bryan roll angle (in degrees).

    Returns:
        H: the hyperrectangle set. =#
    function T_Hyperrectangle(offset::T_RealVector,
                              width::T_Real,
                              height::T_Real,
                              depth::T_Real;
                              yaw::T_Real=0.0,
                              pitch::T_Real=0.0,
                              roll::T_Real=0.0)::T_Hyperrectangle
        if yaw%90!=0 || pitch%90!=0 || roll%90!=0
            err = ArgumentError("ERROR: hyperrectangle must be axis-aligned.")
            throw(err)
        end
        # Compute the hyperrectangle min/max vertices in world frame, no offset
        l = T_RealVector([-width/2, -height/2, 0.0])
        u = T_RealVector([width/2, height/2, depth])
        # Apply rotation
        c = (angle) -> cosd(angle)
        s = (angle) -> sind(angle)
        ψ, θ, φ = yaw, pitch, roll
        Rz = T_RealMatrix([c(ψ) -s(ψ) 0;
                           s(ψ)  c(ψ) 0;
                           0       0  1])
        Ry = T_RealMatrix([ c(θ) 0 s(θ);
                            0    1 0;
                           -s(θ) 0 c(θ)])
        Rx = T_RealMatrix([1 0     0;
                           0 c(φ) -s(φ);
                           0 s(φ)  c(φ)])
        R = Rz*Ry*Rx # **intrinsic** rotations
        lr = R*l
        ur = R*u
        l = min.(lr, ur)
        u = max.(lr, ur)
        # Apply offset
        l += offset
        u += offset
        # Save hyperrectangle
        H = T_Hyperrectangle(l, u)
        return H
    end
end

#= Continuous-time trajectory data structure. =#
struct T_ContinuousTimeTrajectory
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
    function T_ContinuousTimeTrajectory(
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
mutable struct T_Table
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
function T_Table(
    def::Vector{Tuple{T_Symbol, T_String, T_String, T_Int}},
    separator::T_String="|")::T_Table

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

    table = T_Table(headings, sorting, fmt, row, head_print, colw)

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
