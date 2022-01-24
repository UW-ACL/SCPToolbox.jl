#= Top-level package.

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

module SCPToolbox

include("utils/utils.jl")
include("parser/parser.jl")
include("solvers/solvers.jl")

# Types
export RealValue, RealVector, RealMatrix, RealArray
export IntRange
export Optional
export Quaternion
export Ellipsoid, project, ∇
export Hyperrectangle
export ContinuousTimeTrajectory
export Homotopy

# Sequential convex programming solvers
export SCvx, GuSTO, PTR
export SCPSolution, SCPHistory, SCPParameters
export SCP_SOLVED,
    SCP_FAILED,
    SCP_SCALING_FAILED,
    SCP_GUESS_PROJECTION_FAILED,
    SCP_BAD_ARGUMENT,
    SCP_BAD_PROBLEM

# Optimization problem (low-level conic form parser)
export ConicProgram
export @new_variable, @new_parameter, @add_constraint, @add_cost, @value, @jacobian, @scale
export ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
export value, constraints, variables, parameters, cost, jump_model, objective_value
export solve!

# Trajectory problem high-level definition
export TrajectoryProblem
export problem_set_dims!,
    problem_advise_scale!,
    problem_set_integration_action!,
    problem_set_guess!,
    problem_set_callback!,
    problem_set_terminal_cost!,
    problem_set_running_cost!,
    problem_set_dynamics!,
    problem_set_X!,
    problem_set_U!,
    problem_set_s!,
    problem_set_bc!,
    problem_add_table_column!,
    define_conic_constraint!
export FOH, IMPULSE

# Computational functions
export rk4
export linterp,
    zohinterp,
    diracinterp,
    straightline_interpolate,
    logsumexp,
    or,
    squeeze,
    convert_units,
    golden,
    c2d,
    trapz,
    homtransf,
    hominv,
    homdisp,
    homrot,
    scalarize
export rpy, slerp_interpolate, Log, skew, rotate, ddq
export sample

# Plotting and printing
export plot_timeseries_bound!,
    plot_ellipsoids!,
    plot_prisms!,
    plot_convergence,
    setup_axis!,
    generate_colormap,
    rgb,
    rgb2pyplot,
    set_axis_equal,
    create_figure,
    save_figure,
    darken_color
export test_heading
export Yellow, Red, Blue, DarkBlue, Green

### Utilities

import .Utils: Types.RealTypes, Types.RealVector, Types.RealMatrix, Types.RealArray
import .Utils: Types.IntRange
import .Utils: Types.Optional
import .Utils: Quaternion, rpy, slerp_interpolate, Log, skew, rotate, ddq
import .Utils: Ellipsoid, project, ∇
import .Utils: Hyperrectangle
const RealValue = Utils.Types.RealTypes

import .Utils: rk4
import .Utils:
    linterp,
    zohinterp,
    diracinterp,
    straightline_interpolate,
    logsumexp,
    or,
    squeeze,
    convert_units,
    golden,
    c2d,
    trapz,
    homtransf,
    hominv,
    homdisp,
    homrot,
    scalarize
import .Utils: ContinuousTimeTrajectory, sample
import .Utils: Homotopy

import .Utils:
    plot_timeseries_bound!,
    plot_ellipsoids!,
    plot_prisms!,
    plot_convergence,
    setup_axis!,
    generate_colormap,
    rgb,
    rgb2pyplot,
    set_axis_equal,
    create_figure,
    save_figure,
    darken_color
import .Utils: test_heading
import .Utils: Yellow, Red, Blue, DarkBlue, Green

### Low-level conic optimization problem parser

import .Parser: ConicProgram
import .Parser:
    @new_variable, @new_parameter, @add_constraint, @add_cost, @value, @jacobian, @scale
import .Parser: ZERO, NONPOS, L1, SOC, LINF, GEOM, EXP
import .Parser: value, constraints, variables, parameters, cost, jump_model, objective_value
import .Parser: solve!

### Trajectory problem definition

import .Parser: TrajectoryProblem
import .Parser:
    problem_set_dims!,
    problem_advise_scale!,
    problem_set_integration_action!,
    problem_set_guess!,
    problem_set_callback!,
    problem_set_terminal_cost!,
    problem_set_running_cost!,
    problem_set_dynamics!,
    problem_set_X!,
    problem_set_U!,
    problem_set_s!,
    problem_set_bc!,
    problem_add_table_column!,
    define_conic_constraint!
import .Parser: FOH, IMPULSE

### Sequential convex programming solvers

import .Solvers: SCvx, GuSTO, PTR
import .Solvers: SCPSolution, SCPHistory, SCPParameters
import .Solvers:
    SCP_SOLVED,
    SCP_FAILED,
    SCP_SCALING_FAILED,
    SCP_GUESS_PROJECTION_FAILED,
    SCP_BAD_ARGUMENT,
    SCP_BAD_PROBLEM

end
