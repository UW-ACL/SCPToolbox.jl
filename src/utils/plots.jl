#= Plotting functions for trajectory visualization.

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
using Statistics
using Printf
using PyPlot
using Colors

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

# ..:: Globals ::..

const OptPyObject = Union{PyPlot.PyObject,Nothing}

# ..:: Methods ::..

"""
    plot_timeseries_bound!(ax, x_min, x_max, y_bnd, height[; lw])

Plot a constant y-value bound keep-out zone of a timeseries. Supposedly to show
a minimum or a maximum of a quantity on a time history plot.

# Arguments
- `ax`: the figure axis object.
- `x_min`: the left-most value.
- `x_max`: the right-most value.
- `y_bnd`: the bound value, or a vector of values (in which it is assumed that
  the vector components are evenly spaced).
- `height`: the "thickness" of the keep-out slab on the plot.

# Keywords
- `lw`: (optional) the line width for the border.
- `abs`: (optional) toggle using `height` as a relative height above or below
  `y_bnd` (when false) or as an absolute value setting (when true).
"""
function plot_timeseries_bound!(
    ax::PyPlot.PyObject,
    x_min::Types.RealTypes,
    x_max::Types.RealTypes,
    y_bnd::Union{Types.RealTypes,Types.RealVector},
    height::Types.RealTypes;
    lw::Types.RealTypes = 1.75,
    abs::Bool = false,
)::Nothing

    if !abs
        y_max = maximum(y_bnd)
        y_min = minimum(y_bnd)
        y_other = (height >= 0) ? y_max + height : y_min + height
    else
        y_other = height
    end

    if y_bnd isa Array
        x_bnd = LinRange(x_min, x_max, length(y_bnd))
        y_other = repeat([y_other], length(y_bnd))
        x = vcat(x_bnd, reverse(x_bnd), x_bnd[1])
        y = vcat(y_bnd, y_other, y_bnd[1])
        x_bndry = x_bnd
        y_bndry = y_bnd
    else
        x = [x_min, x_max, x_max, x_min, x_min]
        y = [y_bnd, y_bnd, y_other, y_other, y_bnd]
        x_bndry = [x_min, x_max]
        y_bndry = [y_bnd, y_bnd]
    end

    fc = parse(RGB, "#db6245")
    ax.fill(x, y, facecolor = rgb2pyplot(fc, a = 0.5), edgecolor = "none")

    ax.plot(
        x_bndry,
        y_bndry,
        color = "#db6245",
        linewidth = lw,
        linestyle = "--",
        dashes = (2, 3),
        solid_capstyle = "round",
        dash_capstyle = "round",
    )

    return nothing
end

"""
    plot_ellipsoids!(ax, E[, axes][; label])

Draw ellipsoids on the currently active plot.

# Arguments
- `ax`: the figure axis object.
- `E`: array of ellipsoids.
- `axes`: (optional) which 2 axes to project onto.

# Keywords
- `label`: (optional) legend label.
"""
function plot_ellipsoids!(
    ax::PyPlot.PyObject,
    E::Vector{Ellipsoid},
    axes::Types.IntVector = [1, 2];
    label::Union{String,Nothing} = nothing,
)::Nothing
    θ = LinRange(0.0, 2 * pi, 100)
    circle = hcat(cos.(θ), sin.(θ))'
    for i = 1:length(E)
        Ep = project(E[i], axes)
        vertices = Ep.H \ circle .+ Ep.c
        x, y = vertices[1, :], vertices[2, :]
        fc = parse(RGB, "#db6245")
        ax.fill(
            x,
            y,
            facecolor = rgb2pyplot(fc, a = 0.5),
            edgecolor = "#26415d",
            linewidth = 1,
            label = (i == 1) ? label : nothing,
        )
    end
    return nothing
end

"""
    plot_prisms!(ax, H[, axes][; label])

Draw rectangular prisms on the current active plot.

# Arguments
- `ax`: the figure axis object.
- `H`: array of 3D hyperrectangle sets.
- `axes`: (optional) which 2 axes to project onto.

# Keywords
- `label`: (optional) legend label.
"""
function plot_prisms!(
    ax::PyPlot.PyObject,
    H::Vector{Hyperrectangle},
    axes::Types.IntVector = [1, 2];
    label::Union{String,Nothing} = nothing,
)::Nothing
    for i = 1:length(H)
        Hi = H[i]
        x, y = axes
        vertices = RealMatrix(
            [
                Hi.l[x] Hi.u[x] Hi.u[x] Hi.l[x] Hi.l[x]
                Hi.l[y] Hi.l[y] Hi.u[y] Hi.u[y] Hi.l[y]
            ],
        )
        x, y = vertices[1, :], vertices[2, :]
        fc = parse(RGB, "#5da9a1")
        ax.fill(
            x,
            y,
            linewidth = 1,
            facecolor = rgb2pyplot(fc, a = 0.5),
            edgecolor = "#427d77",
            label = (i == 1) ? label : nothing,
        )
    end
    return nothing
end

"""
    plot_convergence(history, name[; xlabel, options, horizontal])

Optimization algorithm convergence and performance plot.

# Arguments
- `history`: SCP iteration data history.
- `name`: the example name.

# Keywords
- `xlabel`: (optional) optional extra string to append to iteration axis label.
- `options`: (optional) pass custom options to `rcParams` that override the
  default ones set in this function.
- `horizontal`: (optional) whether to layout the subplot in a row (by default,
  layout is in a column).
"""
function plot_convergence(
    history,
    name::String;
    xlabel::Optional{String} = nothing,
    options::Dict = Dict(),
    horizontal::Bool = false,
)::Nothing

    plot_statistics = history isa AbstractArray
    if plot_statistics
        all_histories = history
        history = history[end]
    end

    # Common values
    algo = history.subproblems[1].algo
    clr = rgb(generate_colormap(), 1.0)

    # Compute concatenated solution vectors at each iteration
    num_iter = length(history.subproblems)
    xd = [vec(history.subproblems[i].sol.xd) for i = 1:num_iter]
    ud = [vec(history.subproblems[i].sol.ud) for i = 1:num_iter]
    p = [history.subproblems[i].sol.p for i = 1:num_iter]
    Nnx = length(xd[1])
    Nnu = length(ud[1])
    np = length(p[1])
    X = RealMatrix(undef, Nnx + Nnu + np, num_iter)
    for i = 1:num_iter
        X[:, i] = vcat(xd[i], ud[i], p[i])
    end
    DX = RealVector([norm(X[:, i] - X[:, end]) / norm(X[:, end]) for i = 1:(num_iter-1)])
    no_change = findfirst((dx) -> dx == 0, DX)
    if !isnothing(no_change)
        num_iter = no_change - 1
        DX = DX[1:num_iter]
    else
        num_iter -= 1
    end
    iters = Types.IntVector(1:num_iter)

    fig = create_figure(horizontal ? (11, 4) : (5, 6), options = options)

    if num_iter <= 15
        xticks = iters
    else
        step = Int(ceil(num_iter / 15))
        xticks = Types.IntVector(1:step:num_iter)
    end

    # ..:: Convergence plot ::..

    # Restore the array of histories
    if plot_statistics
        trials = all_histories[2:end]
    else
        # Make it an array so we can write general code below
        trials = [history]
    end

    ax = fig.add_subplot(horizontal ? 121 : 211)

    ax.set_yscale("log")
    ax.grid(linewidth = 0.3, alpha = 0.5, axis = "y", which = "major")
    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight = true, axis = "x")
    ax.margins(x = 0.04, y = 0.04)
    ax.set_xticks(xticks)

    ax.set_xlabel("Iteration number" * (isnothing(xlabel) ? "" : ", " * xlabel))
    ax.set_ylabel(
        string(
            "Distance from solution, ",
            "\$\\frac{\\|X_{(\\ell)}-X^*\\|_2}{",
            "\\|X^*\\|_2}\$",
        ),
    )

    ax.plot(
        iters,
        DX,
        color = clr,
        linewidth = 2,
        marker = "o",
        markersize = 6,
        markeredgewidth = 0,
        clip_on = false,
        zorder = 100,
    )

    # Set y ticks
    # Based on: https://stackoverflow.com/a/64840431
    y_major = matplotlib.ticker.LogLocator(base = 10.0, numticks = 5)
    ax.yaxis.set_major_locator(y_major)
    y_minor = matplotlib.ticker.LogLocator(base = 10.0, subs = (1:10) * 0.1, numticks = 100)
    ax.yaxis.set_minor_locator(y_minor)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax_top = ax

    # ..:: Timing performance plot ::..

    ax = fig.add_subplot(horizontal ? 122 : 212)

    ax.set_axisbelow(true)
    ax.set_facecolor("white")
    ax.autoscale(tight = true, axis = "x")
    ax.margins(x = 0.04, y = 0.04)
    ax.set_xticks(xticks)

    ax.set_xlabel("Iteration number")
    ax.set_ylabel("Time per iteration [s]")

    labels = iters
    width = 0.3
    lw = 3.0 + (num_iter - 5) / 35 * (1.5 - 3.0) # Scaling that works visually well
    js = "bevel"
    darken_factor = 0.3
    darken =
        (c) -> rgb2pyplot(
            weighted_color_mean(1 - darken_factor, parse(RGB, c), colorant"black"),
        )

    num_scp_iters = length(trials[1].subproblems) - 1
    discretize = map(
        i -> [trial.subproblems[i].timing[:discretize] for trial in trials],
        1:num_scp_iters,
    )
    solve =
        map(i -> [trial.subproblems[i].timing[:solve] for trial in trials], 1:num_scp_iters)
    formulate = map(
        i -> [trial.subproblems[i].timing[:formulate] for trial in trials],
        1:num_scp_iters,
    )
    overhead = map(
        i -> [trial.subproblems[i].timing[:overhead] for trial in trials],
        1:num_scp_iters,
    )
    total = map(
        trial -> cumsum([
            trial.subproblems[i].timing[:discretize] +
            trial.subproblems[i].timing[:solve] +
            trial.subproblems[i].timing[:formulate] +
            trial.subproblems[i].timing[:overhead]
            # trial.subproblems[i].timing[:total]
            for i = 1:num_scp_iters
        ]),
        trials,
    )

    av_discretize = map(median, discretize)
    av_solve = map(median, solve)
    av_formulate = map(median, formulate)
    av_overhead = map(median, overhead)

    total = hcat(total...)
    av_total = map(iter -> median(total[iter, :]), 1:num_scp_iters)
    min_total = map(iter -> quantile(total[iter, :], 0.1), 1:num_scp_iters)
    max_total = map(iter -> quantile(total[iter, :], 0.9), 1:num_scp_iters)

    ymax = maximum((av_formulate + av_discretize + av_solve + av_overhead)) * 1.1

    for i = 1:2
        linew = (i == 1) ? 0 : lw
        z = (i == 1) ? 0 : -1
        lbl = (str) -> (i == 1) ? str : nothing

        ax.bar(
            labels,
            av_formulate,
            width,
            label = lbl("Formulate"),
            color = DarkBlue,
            linewidth = linew,
            edgecolor = darken(DarkBlue),
            joinstyle = js,
            zorder = z,
        )
        ax.bar(
            labels,
            av_solve,
            width,
            bottom = av_formulate,
            label = lbl("Solve"),
            color = Red,
            linewidth = linew,
            edgecolor = darken(Red),
            joinstyle = js,
            zorder = z,
        )
        ax.bar(
            labels,
            av_discretize,
            width,
            bottom = av_formulate + av_solve,
            label = lbl("Discretize"),
            color = Yellow,
            linewidth = linew,
            edgecolor = darken(Yellow),
            joinstyle = js,
            zorder = z,
        )
        ax.bar(
            labels,
            av_overhead,
            width,
            bottom = av_formulate + av_solve + av_discretize,
            label = lbl("Overhead"),
            color = Green,
            linewidth = linew,
            edgecolor = darken(Green),
            joinstyle = js,
            zorder = z,
        )
    end

    ax.set_ylim(top = ymax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        reverse(handles),
        reverse(labels),
        framealpha = 0.8,
        fontsize = 8,
        loc = "upper left",
    )

    ax2 = ax.twinx()

    ax2_clr = Blue
    outline_w = 1.5

    ax2.set_ylabel("Cumulative time [s]", color = ax2_clr)
    ax2.tick_params(axis = "y", colors = ax2_clr)
    ax2.spines."right".set_edgecolor(ax2_clr)

    # Plots average
    ax2.plot(
        iters,
        av_total,
        color = ax2_clr,
        linewidth = 2,
        marker = "o",
        markersize = 6,
        markeredgewidth = 0,
        markeredgecolor = "white",
        clip_on = false,
        zorder = 100,
    )

    ax2.plot(
        iters,
        av_total,
        color = "white",
        linewidth = 2 + outline_w,
        marker = "o",
        markersize = 6,
        markeredgewidth = outline_w,
        markeredgecolor = "white",
        clip_on = false,
        zorder = 90,
    )

    # Plot error bars
    error_bars = Vector{Matrix{Float64}}(undef, 0)
    for iter = 1:num_scp_iters
        error_bar = hcat([iter; min_total[iter]], [iter; max_total[iter]])
        push!(error_bars, error_bar')
    end
    errors = PyPlot.matplotlib.collections.LineCollection(
        error_bars,
        zorder = 95,
        colors = ax2_clr,
        linewidths = 2,
        capstyle = "round",
    )
    ax2.add_collection(errors)

    error_outlines = PyPlot.matplotlib.collections.LineCollection(
        error_bars,
        zorder = 94,
        colors = "white",
        linewidths = 2 + outline_w,
        capstyle = "round",
    )
    ax2.add_collection(error_outlines)

    ax2.set_ylim(nothing, max_total[end] * 1.05)

    ax_bot = ax

    fig.align_ylabels([ax_top, ax_bot])

    save_figure(@sprintf("%s_convergence.pdf", name), algo)

    return nothing
end

"""
    setup_axis!(ax, rows, cols, k, ijk[; xlabel, ylabel, clabel, tight,
                axis, cbar, cbar_aspect])

Create the standard 2D axis. Note that at one of the following combination must
be entered:
- nothing (a single 111 subplot is generated)
- ax (an existing axis)
- rows, cols, k (subplot specification)
- ijk (subplot specification or gridspec element)

# Arguments
- `ax`: an existing axis object.
- `rows`: the number of subplot rows.
- `cols`: the number of subplot columns.
- `k`: the subplot number.
- `ijk`: k-th subplot in a grid of (i, j) subplots.
- `ijk`: a gridspec element.

# Keywords
- `xlabel`: (optional) the x-axis label.
- `ylabel`: (optional) the y-axis label.
- `clabel`: (optional) colorbar label.
- `tight`: (optional) which axes to make autoscale tight.
- `axis`: (optional) axis scaling.
- `cbar`: (optional) colorbar colormap.
- `cbar_aspect`: (optional) colorbar aspect ratio.

# Returns
- `ax`: the axis object.
"""
function setup_axis!(
    ax::OptPyObject,
    rows::Union{Int,Nothing},
    cols::Union{Int,Nothing},
    k::Union{Int,Nothing},
    ijk::Union{Int,PyPlot.PyObject,Nothing};
    xlabel::Union{String,Nothing} = nothing,
    ylabel::Union{String,Nothing} = nothing,
    clabel::Union{String,Nothing} = nothing,
    tight::String = "",
    axis::Union{String,Nothing} = nothing,
    cbar::OptPyObject = nothing,
    cbar_aspect::Union{Int,Nothing} = nothing,
)::PyPlot.PyObject

    if isnothing(ax)
        if !isnothing(rows)
            # Generate k-th subplot in a grid of (rows, cols) subplots
            ax = plt.gcf().add_subplot(rows, cols, k)
        elseif !isnothing(ijk)
            # Generate k-th subplot in a grid of (i, j) subplots
            ax = plt.gcf().add_subplot(ijk)
        else
            # Generate a single 111 subplot
            ax = plt.gcf().add_subplot()
        end
    end

    ax.grid(linewidth = 0.3, alpha = 0.5)
    ax.set_axisbelow(true)
    ax.set_facecolor("white")

    if !isnothing(axis)
        ax.axis(axis)
    end

    if !isempty(tight)
        ax.autoscale(tight = true, axis = tight)
    end

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if !isnothing(cbar)
        plt.gcf().colorbar(cbar, aspect = cbar_aspect, label = clabel, ax = ax)
    end

    return ax
end

"""
    setup_axis!([; ...])

See the documentation of setup_axis! core function. This is a variant of the
core function which is to be used when you want a single 111 subplot to be
generated.
"""
function setup_axis!(; kwargs...)::PyPlot.PyObject
    return setup_axis!(nothing, nothing, nothing, nothing, nothing; kwargs...)
end

"""
    setup_axis!([; ...])

See the documentation of setup_axis! core function. This is a variant of the
core function which is to be used when you want to provide an existing axis
handle or a gridspec element.
"""
function setup_axis!(ax_or_gspec::PyPlot.PyObject; kwargs...)::PyPlot.PyObject
    try
        # Try as an axis
        return setup_axis!(ax_or_gspec, nothing, nothing, nothing, nothing; kwargs...)
    catch
        # Try as a gridspec element
        try
            return setup_axis!(nothing, nothing, nothing, nothing, ax_or_gspec; kwargs...)
        catch e
            # Some other issue than type ambiguity is going on
            rethrow(e)
        end
    end
end

"""
    setup_axis!([; ...])

See the documentation of setup_axis! core function. This is a variant of the
core function which is to be used when you want to create a k-th axis on a grid
of (rows, cols) subplots.
"""
function setup_axis!(rows::Int, cols::Int, k::Int; kwargs...)::PyPlot.PyObject
    return setup_axis!(nothing, rows, cols, k, nothing; kwargs...)
end

"""
    setup_axis!([; ...])

See the documentation of setup_axis! core function. This is a variant of the
core function which is to be used when you want to create a k-th axis on a grid
of (i rows, j columns) subplots.
"""
function setup_axis!(ijk::Int; kwargs...)::PyPlot.PyObject
    return setup_axis!(nothing, nothing, nothing, nothing, ijk; kwargs...)
end

"""
    generate_colormap([style][; minval, maxval, midval])

Get a plotting colormap.

# Arguments
- `style`: (optional) which PyPlot colormap to use.

# Keywords
- `minval`: (optional) the value to map to the lowest color.
- `maxval`: (optional) the value to map to the highest color.
- `midval`: (optional) the value to map to the middle color.

# Returns
- `cmap`: a colormap object.
"""
function generate_colormap(
    style::String = "inferno_r";
    minval::Types.RealTypes = 0.0,
    maxval::Types.RealTypes = 1.0,
    midval::Types.RealTypes = NaN,
)::PyPlot.PyObject
    cmap = plt.get_cmap(style)
    if isnan(midval)
        nrm = matplotlib.colors.Normalize(vmin = minval, vmax = maxval)
    else
        nrm = matplotlib.colors.TwoSlopeNorm(midval, vmin = minval, vmax = maxval)
    end
    cmap = matplotlib.cm.ScalarMappable(norm = nrm, cmap = cmap)
    return cmap
end

"""
    rgb(cmap, v)

Sample a colormap for an RGB color at given value.

# Arguments
- `cmap`: the colormap object.
- `v`: the value at which to sample the colormap.

# Returns
- `clr`: the RGB color 3-tuple.
"""
function rgb(
    cmap::PyPlot.PyObject,
    v::Types.RealTypes,
)::Tuple{Types.RealTypes,Types.RealTypes,Types.RealTypes}
    clr = cmap.to_rgba(v)[1:3]
    return clr
end

"""
    rgb2pyplot(c, a)

Convert RGB color object to a tuple that PyPlot accepts.

# Arguments
- `c`: the RGB color object.
- `a`: (optional) the alpha (opacity) channel value.

# Returns
- `t`: a 4-tuple (R, G, B, A) for PyPlot.
"""
function rgb2pyplot(c::T; a::Real = 1)::NTuple{4,Real} where {T<:RGB}

    r, g, b = Float64(c.r), Float64(c.g), Float64(c.b)
    t = (r, g, b, a)

    return t
end

"""
    darken_color(clr, amount)

Inteprolate the color closer to black, for higher settings of `amount`.

# Arguments
- `clr`: the original color.
- `amount`: a value between 0 and 1, with 0 returning the original color and 1
  returning the darkest color (black).

# Returns
- `rgba`: the new color as a `(r,g,b,a)` tuple of floats.
"""
function darken_color(clr::String, amount::Float64)::NTuple{4,Real}
    dark_clr = colorant"black"
    new_clr = weighted_color_mean(1 - amount, parse(RGB, clr), dark_clr)
    rgba = rgb2pyplot(new_clr)
    return rgba
end

"""
    set_axis_equal(ax, lims)

Set axis limits with an equal aspect ratio (i.e. circles appear as circles).

# Arguments
- `ax`: the axis object.
- `lims`: a four-tuple of scalars (xmin,xmax,ymin,ymax). Exactly one of these
  has to be `missing`, which means the bound of that limit is decided by the
  scaling amount.
"""
function set_axis_equal(
    ax::PyPlot.PyObject,
    lims::Tuple{
        Union{Real,Missing},
        Union{Real,Missing},
        Union{Real,Missing},
        Union{Real,Missing},
    },
)::Nothing

    ax.axis("equal")
    save_figure("scp_tmp_fig.pdf", "", tmp = true)
    x_rng = ax.get_xlim()
    y_rng = ax.get_ylim()
    ar = (y_rng[2] - y_rng[1]) / (x_rng[2] - x_rng[1])

    ax.axis("auto")
    xmin, xmax, ymin, ymax = lims
    if ismissing(xmin)
        xmin = xmax - (ymax - ymin) / ar
    elseif ismissing(xmax)
        xmax = xmin + (ymax - ymin) / ar
    elseif ismissing(ymin)
        ymin = ymax - ar * (xmax - xmin)
    elseif ismissing(ymax)
        ymax = ymin + ar * (xmax - xmin)
    end
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    return nothing
end

"""
    create_figure(size[; options])

Create an empty figure for plotting.

# Arguments
- `size`: the figure size (width, height).

# Keywords
- `options`: pass custom options to `rcParams` that override the default ones
  set in this function.

# Returns
- `fig`: the figure object.
"""
function create_figure(
    size::Tuple{T,V};
    options::Dict = Dict(),
)::Figure where {T<:Real,V<:Real}

    # Set plot parameters
    rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
    set_with_default(key, val) = begin
        rcParams[key] = (key in keys(options)) ? options[key] : val
    end
    set_with_default("text.usetex", true)
    set_with_default("font.family", "sans-serif")
    set_with_default("axes.labelsize", 14)
    set_with_default("axes.titlesize", 14)
    set_with_default("xtick.labelsize", 12)
    set_with_default("ytick.labelsize", 12)
    set_with_default("text.latex.preamble", string("\\usepackage{sansmath}", "\\sansmath"))

    plt.ioff()

    fig = plt.figure(figsize = size)

    plt.clf()

    return fig
end

"""
    save_figure(filename[, algo][; tmp, path, tight_layout, facecolor, dpi])

Save the current figure to file. The filename is prepended with the name of the
SCP algorithm used for the solution, if provided.

# Arguments
- `filename`: the filename of the figure.
- `algo`: (optioanl) the SCP algorithm string (format "<SCP_ALGO> (backend:
  <CVX_ALGO>)"). If not provided, then do not specify an algorithm name for the
  file.

# Keywords
- `tmp`: (optional) whether this is a temporary file.
- `path`: (optional) where to save the figure.
- `tight_layout`: (optional) apply tight layout to the subplot arrangement.
- `facecolor`: (optional) color of the background (transparent by default).
- `dpi`: (optional) dots per inch, relevant for raster image formats like PNG.
"""
tight_layout_applied = false
function save_figure(
    filename::String,
    algo::String = "";
    tmp::Bool = false,
    path::Union{String,Nothing} = nothing,
    tight_layout::Bool = true,
    facecolor = zeros(4),
    dpi::Int = 300,
)::Nothing

    global tight_layout_applied

    algo = lowercase(split(algo, " "; limit = 2)[1])
    path = isnothing(path) ? joinpath(@__DIR__, "../../media/tests/") : path

    # Apply tight layout, only do this **once** per figure
    if tight_layout && !tight_layout_applied
        plt.tight_layout()
        tight_layout_applied = true
    end

    if !isempty(algo)
        filedir = split(filename, "/")
        filedir[end] = @sprintf("%s_%s", algo, filedir[end])
        filename = string(filedir...)
    end

    # Save figure
    if !tmp
        plt.savefig(
            @sprintf("%s%s", path, filename),
            bbox_inches = "tight",
            pad_inches = 0.01,
            facecolor = facecolor,
            dpi = dpi,
        )
        plt.close()
        tight_layout_applied = false # reset
    else
        filename = split(filename, "/")[end] # Get just the name
        if Sys.iswindows()
             plt.savefig(
                "C:\\Users\\" * ENV["USERNAME"] * "\\AppData\\Local\\Temp\\" * filename,
                bbox_inches = "tight",
                pad_inches = 0.01,
                facecolor = facecolor,
                dpi = dpi,
            )
        else
            plt.savefig(
                @sprintf("/tmp/%s", filename),
                bbox_inches = "tight",
                pad_inches = 0.01,
                facecolor = facecolor,
                dpi = dpi,
            )
        end
    end

    return nothing
end
