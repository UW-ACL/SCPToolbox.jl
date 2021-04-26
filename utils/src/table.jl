#= Iteration progress table type.

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
    include("basic_types.jl")
end

using Printf

import Base: print, reset

export Table, improvement_percent

""" Iteration progress information table to be printed in REPL. """
mutable struct Table
    headings::Array{String}    # Column headings
    sorting::Dict{Symbol, Int} # Column order
    fmt::Dict{Symbol, String}  # Column value format
    row::String                # The full row format
    # >> Private members <<
    __head_print::Bool # Flag whether the column headings have been printed
    __colw::IntVector  # Column widths

    """
        Table(def, separator)

    Table constructor.

    # Arguments
    - `def`: an array of column definitions, where each element is a tuple of:
      - `[1]`: unique symbol referencing this colum.
      - `[2]`: column heading.
      - `[3]`: column format (for sprintf to convert into a string).
      - `[4]`: column width.

    # Returns
    - `table`: the table structure.
    """
    function Table(
        def::Vector{Tuple{Symbol, String, String, Int}},
        separator::String="|")::Table

        # Initialize
        headings = Array{String}(undef, 0)
        sorting = Dict{Symbol, Int}()
        fmt = Dict{Symbol, String}()
        row = ""
        colw = IntVector(undef, 0)
        args = (headings, sorting, fmt, colw, separator)

        # Add each column
        for d in def
            sym, head, n2s, w = d[1], d[2], d[3], d[4]
            row = add_table_column!(sym, head, n2s, w, args..., row)
        end

        # Default values
        head_print = true

        table = new(headings, sorting, fmt, row, head_print, colw)

        return table
    end # function
end # struct

"""
    add_table_column!(col_sym, col_heading, col_fmt, col_width, headings,
                      sorting, fmt, colw, sep, rowg)

Add a new column to the iteration info table.

# Arguments
- `col_sym`: symbol that references this column.
- `col_heading`: the column heading.
- `col_fmt`: the column format (for sprintf to convert from native to string).
- `col_width`: the column's width.
- `headings`: the existing array of table column headings.
- `sorting`: the column sort order, so that we can re-define column ordering
  seamlessly.
- `colw`: vector of column widths.
- `fmt`: the existing string of table column format.
- `sep`: table column separator.
- `row`: the complete row format specification to add this new column to.

# Returns
- `fmt`: the updated string of table column format.
"""
function add_table_column!(
    col_sym::Symbol,
    col_heading::String,
    col_fmt::String,
    col_width::Int,
    headings::Vector{String},
    sorting::Dict{Symbol,Int},
    fmt::Dict{Symbol,String},
    colw::IntVector,
    sep::String,
    row::String)::String

    # Column separator
    separator = (length(row)==0) ? "" : string(" ", sep, " ")

    push!(headings, col_heading)
    push!(sorting, col_sym => length(headings))
    push!(fmt, col_sym => col_fmt)
    push!(colw, col_width)

    row = string(row, separator, "%-", col_width, "s")

    return row
end # function

"""
    print_hrule(table)

Print table row horizontal separator line.

# Arguments
- `table`: the table specification.
"""
function print_hrule(table::Table)::Nothing
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
end # function

"""
    reset(table)

Reset table printing. This will make the column headings be printed again.

# Arguments
- `table`: the table specification.
"""
function reset(table::Table)::Nothing
    table.__head_print = true
    return nothing
end

"""
    print(row, table)

Print row of table.

# Arguments
- `row`: table row specification.
- `table`: the table specification.
"""
function print(row::Dict{Symbol, T}, table::Table)::Nothing where {T}
    # Assign values to table columns
    values = fill("", length(table.headings))
    for (k, v) in row
        val_fmt = table.fmt[k]
        values[table.sorting[k]] = @eval @sprintf($val_fmt, $v)
    end

    if table.__head_print==true
        table.__head_print = false
        # Print the columnd headers
        @eval @printf($(table.row), $(table.headings)...)
        println()

        print_hrule(table)
    end

    @eval @printf($(table.row), $values...)
    println()

    return nothing
end

"""
    improvement_percent(J_new, J_old)

Compute the relative cost improvement (as a string to be put into a table).

# Arguments
- `J_new`: next cost.
- `J_old`: old cost.

# Returns
- `ΔJ`: the relative cost improvement.
"""
function improvement_percent(J_new::RealValue,
                             J_old::RealValue)::String
    if isnan(J_old)
        ΔJ = ""
    else
        ΔJ = (J_old-J_new)/abs(J_old)*100
        _ΔJ = @sprintf("%.2f", ΔJ)
        if length(_ΔJ)>8
            fmt = string("%.", (ΔJ>0) ? 2 : 1, "e")
            ΔJ = @eval @sprintf($fmt, $ΔJ)
        else
            ΔJ = _ΔJ
        end
    end

    return ΔJ
end
