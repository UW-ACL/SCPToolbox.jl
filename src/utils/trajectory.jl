#= Continuous-time trajectory type.

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

import ..linterp, ..zohinterp, ..diracinterp

export ContinuousTimeTrajectory, sample

""" Continuous-time trajectory data structure.

Possible interpolation methods are:
- `:linear` (linear interpolation);
- `:zoh` (zeroth-order hold interpolation).
- `:impulse` (impulse signal interpolation).
"""
struct ContinuousTimeTrajectory
    t::RealVector  # The trajectory time nodes
    x::RealArray   # The trajectory values at the corresponding times
    interp::Symbol # Interpolation between time nodes

    """ Constructor.

    # Arguments
        t: the trajectory time nodes.
        x: the trajectory values at the corresponding times.
        interp: the interpolation method.

    # Returns
        traj: the continuous-time trajectory.
    """
    function ContinuousTimeTrajectory(
        t::RealVector,
        x::RealArray,
        interp::Symbol)::ContinuousTimeTrajectory

        if !(interp in [:linear, :zoh, :impulse])
            err = ArgumentError("unknown trajectory interpolation type.")
            throw(err)
        end

        traj = new(t, x, interp)

        return traj
    end
end # struct

""" Get the value of a continuous-time trajectory at time t.

# Arguments
- `traj`: the trajectory.
- `t`: the evaluation time.

# Returns
- `x`: the trajectory value at time t.
"""
function sample(traj::ContinuousTimeTrajectory,
                t::RealTypes)::Union{RealTypes, RealArray}

    if traj.interp==:linear
        x = linterp(t, traj.x, traj.t)
    elseif traj.interp==:zoh
        x = zohinterp(t, traj.x, traj.t)
    elseif traj.interp==:impulse
        x = diracinterp(t, traj.x, traj.t)
    end

    return x
end
