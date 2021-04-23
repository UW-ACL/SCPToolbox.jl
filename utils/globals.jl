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

using LinearAlgebra
using JuMP
using ECOS
using Printf
using PyPlot
using Colors

# List of possible SCP statuses/errors
@enum(SCPStatus,
      SCP_SOLVED,
      SCP_FAILED,
      SCP_SCALING_FAILED,
      SCP_GUESS_PROJECTION_FAILED,
      SCP_BAD_ARGUMENT,
      SCP_BAD_PROBLEM)

# Colors
const Yellow = "#f1d46a"
const Red = "#db6245"
const Blue = "#356397"
const DarkBlue = "#26415d"
const Green = "#5da9a1"
