# GLMakie rendering and controls for TriangulationCache.

############################
# Color and Legend Helpers
############################

# Return the default colormap.
function default_continuous_colormap()
    return [
        GLMakie.RGBf(0.015, 0.015, 0.035),
        GLMakie.RGBf(0.200, 0.020, 0.250),
        GLMakie.RGBf(0.600, 0.040, 0.180),
        GLMakie.RGBf(0.900, 0.250, 0.070),
        GLMakie.RGBf(0.990, 0.720, 0.200),
        GLMakie.RGBf(1.000, 0.980, 0.760),
    ]
end

# Pad flat color ranges.
function finite_color_range(values::AbstractVector{<:Real})
    lo, hi = extrema(values)
    if lo == hi
        pad = max(abs(lo), 1.0) / 2
        return (lo - pad, hi + pad)
    end
    return (lo, hi)
end

# Build categorical colors.
function categorical_palette(n::Integer)
    base_colors = GLMakie.Makie.wong_colors()
    n <= length(base_colors) && return base_colors[1:n]

    extra_colors = map(GLMakie.Makie.to_color, [:purple, :brown, :pink, :gray, :olive, :cyan])
    colors = vcat(base_colors, extra_colors)
    while length(colors) < n
        append!(colors, colors)
    end
    return colors[1:n]
end

function value_label(value::Real)
    return value == round(value) ? string(Int(round(value))) : string(value)
end

function value_label(value)
    return value isa Real ? value_label(value) : string(value)
end

# Add a value legend.
function add_value_legend!(fig, elements, labels, title)
    isempty(elements) && return nothing
    return GLMakie.Legend(fig[1, 1], elements, labels, title;
        tellwidth=false, tellheight=false, halign=:right, valign=:top,
        margin=(10, 10, 10, 10))
end

############################
# Plot Value Helpers
############################

function triangle_plot_value(TC::TriangulationCache, triangle::Vector{Int64}; plot_log_transform=false)
    vertex_values = non_wildcard_values(function_values(TC)[triangle])
    isempty(vertex_values) && return nothing
    if all(is_real_value, vertex_values)
        value = numeric_mean_or_nothing(vertex_values)
        if plot_log_transform
            value <= -1 && return nothing
            value = log(value + 1)
        end
        return isfinite(value) ? value : nothing
    end

    all(==(first(vertex_values)), vertex_values) || error("Cannot plot a complete triangle with non-equal non-real vertex values: $(vertex_values).")
    return first(vertex_values)
end

function cached_plot_value(value; plot_log_transform=false)
    is_wildcard_value(value) && return nothing
    if value isa Real
        if plot_log_transform
            value <= -1 && return nothing
            value = log(value + 1)
        end
        return isfinite(value) ? value : nothing
    end
    return value
end

function categorical_unique_values_any(values)
    categories = unique(filter(value -> !isnothing(value) && !is_wildcard_value(value), values))
    return sort(categories; by=string)
end

function append_missing_values!(values::Vector{Any}, candidates)
    for value in candidates
        if !isnothing(value) && !is_wildcard_value(value) && !(value in values)
            push!(values, value)
        end
    end
    return values
end

function stable_plot_value_order!(TC::TriangulationCache, visible_values; plot_log_transform=false)
    values = get!(TC.plot_value_order, Bool(plot_log_transform), Any[])
    cached_values = (cached_plot_value(value; plot_log_transform=plot_log_transform) for value in output_values(TC))
    append_missing_values!(values, cached_values)
    append_missing_values!(values, visible_values)
    return values
end

function vertex_plot_values(triangle_values)
    colors = Vector{Any}(undef, 3 * length(triangle_values))
    for (i, value) in pairs(triangle_values)
        colors[(3i - 2):(3i)] .= value
    end
    return colors
end

function continuous_vertex_values(values, categories)
    if all(value -> value === nothing || value isa Real, values)
        return [value === nothing ? NaN : Float64(value) for value in values]
    end
    category_index = Dict(value => Float64(i) for (i, value) in pairs(categories))
    return [value === nothing ? NaN : category_index[value] for value in values]
end

function numeric_continuous_color_range(TC::TriangulationCache, visible_values; plot_log_transform=false)
    cached_values = [
        cached_plot_value(value; plot_log_transform=plot_log_transform)
        for value in output_values(TC)
    ]
    values = [
        Float64(value)
        for value in vcat(cached_values, visible_values)
        if value isa Real && isfinite(value)
    ]
    return isempty(values) ? (0.0, 1.0) : finite_color_range(values)
end

############################
# Mesh Assembly
############################

function selected_triangles(TC::TriangulationCache, plot_all_triangles::Bool)
    triangles = if plot_all_triangles || isempty(complete_triangles(TC))
        vcat(complete_triangles(TC), incomplete_triangles(TC))
    else
        complete_triangles(TC)
    end
    return [T for T in triangles if triangle_intersects_window(TC, T)]
end

function triangulation_mesh_data(TC::TriangulationCache, triangles::TriangleList, triangle_values)
    total_vertices = 3 * length(triangles)
    vertices = Matrix{Float64}(undef, total_vertices, 2)
    faces = Matrix{Int64}(undef, length(triangles), 3)

    for (triangle_index, triangle) in enumerate(triangles)
        vertex_offset = 3 * (triangle_index - 1)
        for local_index in 1:3
            point = get_point(triangulation(TC), triangle[local_index])
            row = vertex_offset + local_index
            vertices[row, 1] = point[1]
            vertices[row, 2] = point[2]
            faces[triangle_index, local_index] = row
        end
    end

    return vertices, faces
end

function triangle_edge_coordinates(vertices, faces)
    xs = Float64[]
    ys = Float64[]
    for face_index in axes(faces, 1)
        for vertex_index in (faces[face_index, 1], faces[face_index, 2], faces[face_index, 3], faces[face_index, 1])
            push!(xs, vertices[vertex_index, 1])
            push!(ys, vertices[vertex_index, 2])
        end
        push!(xs, NaN)
        push!(ys, NaN)
    end
    return xs, ys
end

function add_triangle_edges!(ax, decorations, vertices, faces; color, linewidth)
    edge_xs, edge_ys = triangle_edge_coordinates(vertices, faces)
    push!(decorations, GLMakie.lines!(ax, edge_xs, edge_ys; color=color, linewidth=linewidth))
    return decorations
end

############################
# Drawing Lifecycle
############################

function set_axis_window!(ax, TC::TriangulationCache)
    GLMakie.xlims!(ax, TC.xlims[1], TC.xlims[2])
    GLMakie.ylims!(ax, TC.ylims[1], TC.ylims[2])
    return ax
end

function draw_triangulation!(fig, ax, TC::TriangulationCache; kwargs...)
    plot_log_transform = get(kwargs, :plot_log_transform, false)
    plot_all_triangles = get(kwargs, :plot_all_triangles, !is_discrete(TC))
    colormap = get(kwargs, :colormap, default_continuous_colormap())
    legend_max_values = get(kwargs, :legend_max_values, TRIANGULATION_CACHE_DEFAULT_LEGEND_LIMIT)
    discrete_legend = get(kwargs, :discrete_legend, nothing)
    legend_title = get(kwargs, :legend_title, "value")
    plot_triangle_edges = get(kwargs, :plot_triangle_edges, false)
    triangle_edge_color = get(kwargs, :triangle_edge_color, GLMakie.RGBAf(0, 0, 0, 0.35))
    triangle_edge_linewidth = get(kwargs, :triangle_edge_linewidth, 0.5)
    decorations = Any[]

    triangles = selected_triangles(TC, plot_all_triangles)
    triangle_values = [triangle_plot_value(TC, T; plot_log_transform=plot_log_transform) for T in triangles]
    vertices, faces = triangulation_mesh_data(TC, triangles, triangle_values)
    vertex_values = vertex_plot_values(triangle_values)

    if size(faces, 1) == 0
        return (plot=nothing, decorations=decorations)
    end

    visible_categories = categorical_unique_values_any(vertex_values)
    if isempty(visible_categories)
        plt = GLMakie.mesh!(ax, vertices, faces; color=:black, shading=false)
        plot_triangle_edges && add_triangle_edges!(ax, decorations, vertices, faces; color=triangle_edge_color, linewidth=triangle_edge_linewidth)
        return (plot=plt, decorations=decorations)
    end

    categories = stable_plot_value_order!(TC, vertex_values; plot_log_transform=plot_log_transform)
    use_discrete_legend = discrete_legend === nothing ? 0 < length(categories) <= legend_max_values : discrete_legend
    if use_discrete_legend
        category_colors = categorical_palette(length(categories))
        color_map = Dict(value => color for (value, color) in zip(categories, category_colors))
        mesh_colors = [value === nothing ? GLMakie.RGBAf(0, 0, 0, 1) : color_map[value] for value in vertex_values]
        plt = GLMakie.mesh!(ax, vertices, faces; color=mesh_colors, shading=false)
        plot_triangle_edges && add_triangle_edges!(ax, decorations, vertices, faces; color=triangle_edge_color, linewidth=triangle_edge_linewidth)
        elements = [GLMakie.PolyElement(color=color, strokecolor=color) for color in category_colors]
        labels = value_label.(categories)
        legend = add_value_legend!(fig, elements, labels, legend_title)
        legend === nothing || push!(decorations, legend)
        return (plot=plt, decorations=decorations)
    end

    continuous_values = continuous_vertex_values(vertex_values, categories)
    colorrange = numeric_continuous_color_range(TC, vertex_values; plot_log_transform=plot_log_transform)
    plt = GLMakie.mesh!(ax, vertices, faces; color=continuous_values, colormap=colormap, colorrange=colorrange, nan_color=:black, shading=false)
    plot_triangle_edges && add_triangle_edges!(ax, decorations, vertices, faces; color=triangle_edge_color, linewidth=triangle_edge_linewidth)
    push!(decorations, GLMakie.Colorbar(fig[1, 2], plt))
    return (plot=plt, decorations=decorations)
end

function delete_drawn_triangulation!(ax, drawn)
    drawn.plot === nothing || GLMakie.delete!(ax, drawn.plot)
    for decoration in drawn.decorations
        try
            GLMakie.delete!(decoration)
        catch
        end
    end
    return nothing
end

function redraw_triangulation!(fig, ax, TC::TriangulationCache, drawn_ref; kwargs...)
    set_axis_window!(ax, TC)
    delete_drawn_triangulation!(ax, drawn_ref[])
    drawn_ref[] = draw_triangulation!(fig, ax, TC; kwargs...)
    return drawn_ref[]
end

############################
# Interactive Controls
############################

function max_area_from_slider_exponent(TC::TriangulationCache, exponent::Real)
    return bounding_box_area(TC) * 10.0 ^ (-Int(round(exponent)))
end

function default_max_area_slider_range()
    return 2:6
end

function default_max_area_slider_start(TC::TriangulationCache, slider_range)
    if TC.max_refinement_area === nothing || TC.max_refinement_area <= 0
        return first(slider_range)
    end
    default_start = Int(round(-log10(TC.max_refinement_area / bounding_box_area(TC))))
    return slider_range[argmin(abs.(slider_range .- default_start))]
end

function max_area_slider_label(exponent::Real)
    return "Max area: window * 1e-" * string(Int(round(exponent)))
end

function navigate_and_refine!(
        fig,
        ax,
        TC::TriangulationCache,
        drawn_ref;
        translate=(0.0, 0.0),
        zoom_factor=nothing,
        seed_and_refine=true,
        navigation_initial_resolution=250,
        navigation_refinement_budget=1000,
        verbose=is_verbose(TC),
        kwargs...)

    change_window_and_refine!(
        TC;
        translate=translate,
        zoom_factor=zoom_factor,
        seed_and_refine=seed_and_refine,
        navigation_initial_resolution=navigation_initial_resolution,
        navigation_refinement_budget=navigation_refinement_budget,
        verbose=verbose,
    )
    redraw_triangulation!(fig, ax, TC, drawn_ref; kwargs...)
    return TC
end

function add_refine_button!(
        fig,
        ax,
        TC::TriangulationCache,
        drawn_ref;
        button_refinement_passes=1,
        navigation_step=0.20,
        zoom_step=0.20,
        refine_button_min_area_factor=0.80,
        max_area_refine_controls=true,
        max_area_slider_range=default_max_area_slider_range(),
        max_area_slider_start=nothing,
        navigation_initial_resolution=250,
        navigation_refinement_budget=1000,
        plot_triangle_edges=false,
        triangle_edge_color=GLMakie.RGBAf(0, 0, 0, 0.35),
        triangle_edge_linewidth=0.5,
        verbose=is_verbose(TC),
        kwargs...)

    controls = fig[2, :] = GLMakie.GridLayout(tellwidth=false, tellheight=true)
    refine_button = GLMakie.Button(controls[1, 1]; label="Refine", tellwidth=false, width=120, height=30)
    left_button = GLMakie.Button(controls[1, 2]; label="←", tellwidth=false, width=42, height=30)
    right_button = GLMakie.Button(controls[1, 3]; label="→", tellwidth=false, width=42, height=30)
    up_button = GLMakie.Button(controls[1, 4]; label="↑", tellwidth=false, width=42, height=30)
    down_button = GLMakie.Button(controls[1, 5]; label="↓", tellwidth=false, width=42, height=30)
    zoom_in_button = GLMakie.Button(controls[1, 6]; label="Zoom +", tellwidth=false, width=84, height=30)
    zoom_out_button = GLMakie.Button(controls[1, 7]; label="Zoom -", tellwidth=false, width=84, height=30)
    edge_button_column = max_area_refine_controls ? 11 : 8
    edge_button = GLMakie.Button(controls[1, edge_button_column]; label="Edges", tellwidth=false, width=84, height=30)
    edge_visible = Ref(Bool(plot_triangle_edges))
    GLMakie.rowgap!(fig.layout, 8)
    GLMakie.rowsize!(fig.layout, 2, GLMakie.Fixed(52))
    GLMakie.rowsize!(controls, 1, GLMakie.Fixed(40))

    redraw() = redraw_triangulation!(
        fig,
        ax,
        TC,
        drawn_ref;
        plot_triangle_edges=edge_visible[],
        triangle_edge_color=triangle_edge_color,
        triangle_edge_linewidth=triangle_edge_linewidth,
        kwargs...)

    if max_area_refine_controls
        slider_range = collect(max_area_slider_range)
        isempty(slider_range) && error("max_area_slider_range must contain at least one value.")
        slider_start = max_area_slider_start === nothing ? default_max_area_slider_start(TC, slider_range) : max_area_slider_start
        max_area_slider = GLMakie.Slider(controls[1, 9]; range=slider_range, startvalue=slider_start, tellwidth=true, width=260)
        max_area_label = GLMakie.Label(
            controls[1, 8],
            GLMakie.lift(max_area_slider_label, max_area_slider.value);
            tellwidth=false,
            width=130,
        )
        max_area_button = GLMakie.Button(controls[1, 10]; label="Fully Refine", tellwidth=false, width=120, height=30)

        GLMakie.on(max_area_button.clicks) do _
            refine_to_max_area!(TC, max_area_from_slider_exponent(TC, max_area_slider.value[]); verbose=verbose)
            redraw()
        end
    end

    GLMakie.on(refine_button.clicks) do _
        for _ in 1:button_refinement_passes
            TC.min_refinement_area *= refine_button_min_area_factor
            refine!(TC)
        end
        redraw()
    end

    navigate(; translate=(0.0, 0.0), zoom_factor=nothing, seed_and_refine=true) = navigate_and_refine!(
        fig,
        ax,
        TC,
        drawn_ref;
        translate=translate,
        zoom_factor=zoom_factor,
        seed_and_refine=seed_and_refine,
        navigation_initial_resolution=navigation_initial_resolution,
        navigation_refinement_budget=navigation_refinement_budget,
        verbose=verbose,
        plot_triangle_edges=edge_visible[],
        triangle_edge_color=triangle_edge_color,
        triangle_edge_linewidth=triangle_edge_linewidth,
        kwargs...)

    GLMakie.on(edge_button.clicks) do _
        edge_visible[] = !edge_visible[]
        redraw()
    end

    GLMakie.on(left_button.clicks) do _
        navigate(translate=(-navigation_step, 0.0))
    end
    GLMakie.on(right_button.clicks) do _
        navigate(translate=(navigation_step, 0.0))
    end
    GLMakie.on(up_button.clicks) do _
        navigate(translate=(0.0, navigation_step))
    end
    GLMakie.on(down_button.clicks) do _
        navigate(translate=(0.0, -navigation_step))
    end
    GLMakie.on(zoom_in_button.clicks) do _
        navigate(zoom_factor=1 - zoom_step, seed_and_refine=false)
    end
    GLMakie.on(zoom_out_button.clicks) do _
        navigate(zoom_factor=1 + zoom_step)
    end

    return refine_button
end

############################
# Figure API
############################

"""
    visualize(TC::TriangulationCache; kwargs...) -> GLMakie.Figure

Render a `TriangulationCache` using GLMakie.

Useful keyword arguments:
- `refine_button`: add interactive refinement controls, default `true`.
- `button_refinement_passes`: number of refinement passes per button click.
- `navigation_step`: arrow-button pan amount as a fraction of window size.
- `zoom_step`: zoom amount as a fraction of window size.
- `navigation_refinement_budget`: oracle-call budget after pan/zoom.
- `navigation_initial_resolution`: coarse mesh size seeded after pan/zoom.
- `max_area_refine_controls`: add exponent slider and Fully Refine button.
- `max_area_slider_range`: integer exponents for `window_area * 1e-X`, default `2:6`.
- `figure_size`: Makie figure size, default `(900, 900)`.
- `plot_all_triangles`: include incomplete triangles in the colored mesh.
- `plot_triangle_edges`: overlay thin triangle edges, default `false`.
- `triangle_edge_color`: edge overlay color.
- `triangle_edge_linewidth`: edge overlay line width.
- interactive figures include an Edges button that toggles edge visibility.
- `legend_max_values`: categorical legend threshold, default `20`.
- `discrete_legend`: force or disable categorical legend behavior.
"""
function visualize(TC::TriangulationCache; kwargs...)::GLMakie.Figure
    refine_button = get(kwargs, :refine_button, true)
    button_refinement_passes = get(kwargs, :button_refinement_passes, 1)

    figure_size = get(kwargs, :figure_size, refine_button ? (1300, 900) : (900, 900))
    fig = GLMakie.Figure(size=figure_size)
    ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=GLMakie.DataAspect(), backgroundcolor=:black)
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(1))
    GLMakie.rowsize!(fig.layout, 1, GLMakie.Relative(1))
    set_axis_window!(ax, TC)

    drawn_ref = Ref{Any}(draw_triangulation!(fig, ax, TC; kwargs...))

    refine_button && add_refine_button!(fig, ax, TC, drawn_ref; button_refinement_passes=button_refinement_passes, kwargs...)

    return fig
end

"""
    visualize(function_oracle::Function; kwargs...) -> (TriangulationCache, GLMakie.Figure)

Construct, refine, display, and return a `TriangulationCache` and its Makie
figure. `total_resolution` is interpreted as the total oracle-call budget,
including the initial mesh.
"""
function visualize(function_oracle::Function; total_resolution=nothing, max_refinement_area=nothing, verbose=true, kwargs...)
    total_given = total_resolution !== nothing
    resolved_total = total_given ? Int(total_resolution) : TRIANGULATION_CACHE_DEFAULT_TOTAL_RESOLUTION
    TC = TriangulationCache(function_oracle; resolution=resolved_total, verbose=verbose, kwargs...)
    if max_refinement_area !== nothing
        total_given && println("Warning: max_refinement_area was supplied, so total_resolution is ignored after the initial mesh.")
        refine_to_max_area!(TC, Float64(max_refinement_area); verbose=verbose)
        TC.oracle_budget = nothing
    else
        TC.oracle_budget = max(0, resolved_total - TC.total_oracle_calls)
        refine_until_budget_exhausted!(TC; verbose=verbose)
        TC.oracle_budget = nothing
    end
    fig = visualize(TC; kwargs...)
    display(fig)
    return TC, fig
end

"""
    save(fig::GLMakie.Figure, filename::String; file_extension="png", dpi=300)

Save a GLMakie figure under `OutputFiles/` unless `filename` already starts
with that directory. Returns the final filename.
"""
function save(fig::GLMakie.Figure, filename::String; file_extension = "png", dpi = 300)
    if !startswith(filename, "OutputFiles/")
        filename = "OutputFiles/" * filename
    end
    if !endswith(filename, file_extension)
        filename *= "." * file_extension
    end
    GLMakie.save(filename, fig; px_per_unit=dpi/150)
    println("Plot saved to $filename")
    return filename
end
