#= Global variables and modules used by the code.

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

using Printf

import Base: showerror

export SCPStatus,
    SCP_SOLVED,
    SCP_FAILED,
    SCP_SCALING_FAILED,
    SCP_GUESS_PROJECTION_FAILED,
    SCP_BAD_ARGUMENT,
    SCP_BAD_PROBLEM
export SCPError
export Yellow, Red, Blue, DarkBlue, Green

# List of possible SCP statuses/errors
@enum(
    SCPStatus,
    SCP_SOLVED,
    SCP_FAILED,
    SCP_SCALING_FAILED,
    SCP_GUESS_PROJECTION_FAILED,
    SCP_BAD_ARGUMENT,
    SCP_BAD_PROBLEM
)

# Colors
const Yellow = "#f1d46a"
const Red = "#db6245"
const Blue = "#356397"
const DarkBlue = "#26415d"
const Green = "#5da9a1"

""" Error exception in SCvx algorithm. Hopefully it doesn't happen! """
struct SCPError <: Exception
    k::Int            # At what discrete time step the error occured
    status::SCPStatus # Error status code
    msg::String       # Error message
end

"""
    showerror(io, e)

Pretty print the custom SCP error.

# Arguments
- `io`: the IO stream object.
- `e`: the error object.
"""
function showerror(io::IO, e::SCPError)::Nothing

    @printf(io, "%s (status: %s)", e.msg, e.status)

    return nothing
end
