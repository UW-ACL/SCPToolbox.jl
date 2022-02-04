#= Tests for lossless convexification double integrator.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

using Test

function lcvx()::Nothing

    for param_choice = 1:2
        # @printf("# Parameter set %d/%d\n", param_choice, 2)

        mdl = DoubleIntegratorParameters(param_choice)
        tests_failed = false

        sol_mp = try
            solve_mp(mdl)
        catch
            @test false # Maximum principle shooting failed
            tests_failed = true
        end
        @test true

        # Solve numerically via lossless convexification
        sol_lcvx = try
            solve_lcvx(mdl)
        catch
            @test false # Lossless convexification failed
            tests_failed = true
        end
        @test true

        if !tests_failed
            plot_trajectory(sol_lcvx, sol_mp, param_choice)
        end
    end

    return nothing
end
