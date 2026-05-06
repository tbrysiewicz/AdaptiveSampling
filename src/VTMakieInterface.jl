# GLMakie rendering for ValuedTriangulation.

function polygon_plot_value(VT::ValuedTriangulation, triangle::Vector{Int64}; plot_log_transform=false)
    vertex_values = non_wildcard_values(map(last, function_cache(VT)[triangle]))
    isempty(vertex_values) && return nothing
    value = numeric_mean_or_nothing(vertex_values)
    if value !== nothing
        if plot_log_transform
            value <= -1 && return nothing
            value = log(value + 1)
        end
        return isfinite(value) ? value : nothing
    end

    all(==(first(vertex_values)), vertex_values) || return nothing
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

function stable_plot_value_order!(VT::ValuedTriangulation, visible_values; plot_log_transform=false)
    values = get!(VT.plot_value_order, Bool(plot_log_transform), Any[])
    cached_values = (cached_plot_value(value; plot_log_transform=plot_log_transform) for value in output_values(VT))
    append_missing_values!(values, cached_values)
    append_missing_values!(values, visible_values)
    return values
end

function should_use_discrete_legend_any(values; legend_max_values=VALUED_TRIANGULATION_DEFAULT_LEGEND_LIMIT, discrete_legend=nothing)
    categories = categorical_unique_values_any(values)
    use_legend = discrete_legend === nothing ? 0 < length(categories) <= legend_max_values : discrete_legend
    return use_legend, categories
end

function value_label(value::Real)
    return value == round(value) ? string(Int(round(value))) : string(value)
end

function value_label(value)
    return value isa Real ? value_label(value) : string(value)
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

function numeric_continuous_color_range(values)
    numeric_values = filter(isfinite, values)
    return isempty(numeric_values) ? (0.0, 1.0) : finite_color_range(numeric_values)
end

function numeric_continuous_color_range(VT::ValuedTriangulation, visible_values; plot_log_transform=false)
    cached_values = [
        cached_plot_value(value; plot_log_transform=plot_log_transform)
        for value in output_values(VT)
    ]
    values = [
        Float64(value)
        for value in vcat(cached_values, visible_values)
        if value isa Real && isfinite(value)
    ]
    return isempty(values) ? (0.0, 1.0) : finite_color_range(values)
end

function selected_triangles(VT::ValuedTriangulation, plot_all_triangles::Bool)
    triangles = if plot_all_triangles || isempty(complete_triangles(VT))
        vcat(complete_triangles(VT), incomplete_triangles(VT))
    else
        complete_triangles(VT)
    end
    return [T for T in triangles if triangle_intersects_window(VT, T)]
end

function triangulation_mesh_data(VT::ValuedTriangulation, triangles::PolygonList, triangle_values)
    total_vertices = 3 * length(triangles)
    vertices = Matrix{Float64}(undef, total_vertices, 2)
    faces = Matrix{Int64}(undef, length(triangles), 3)

    for (triangle_index, triangle) in enumerate(triangles)
        vertex_offset = 3 * (triangle_index - 1)
        for local_index in 1:3
            point = function_cache(VT)[triangle[local_index]][1]
            row = vertex_offset + local_index
            vertices[row, 1] = point[1]
            vertices[row, 2] = point[2]
            faces[triangle_index, local_index] = row
        end
    end

    return vertices, faces
end

function draw_triangulation!(fig, ax, VT::ValuedTriangulation; kwargs...)
    plot_log_transform = get(kwargs, :plot_log_transform, false)
    plot_all_triangles = get(kwargs, :plot_all_triangles, !is_discrete(VT))
    colormap = get(kwargs, :colormap, default_continuous_colormap())
    legend_max_values = get(kwargs, :legend_max_values, VALUED_TRIANGULATION_DEFAULT_LEGEND_LIMIT)
    discrete_legend = get(kwargs, :discrete_legend, nothing)
    legend_title = get(kwargs, :legend_title, "value")
    decorations = Any[]

    triangles = selected_triangles(VT, plot_all_triangles)
    triangle_values = [polygon_plot_value(VT, T; plot_log_transform=plot_log_transform) for T in triangles]
    vertices, faces = triangulation_mesh_data(VT, triangles, triangle_values)
    vertex_values = vertex_plot_values(triangle_values)

    if size(faces, 1) == 0
        return (plot=nothing, decorations=decorations)
    end

    visible_categories = categorical_unique_values_any(vertex_values)
    if isempty(visible_categories)
        return (plot=GLMakie.mesh!(ax, vertices, faces; color=:black, shading=false), decorations=decorations)
    end

    categories = stable_plot_value_order!(VT, vertex_values; plot_log_transform=plot_log_transform)
    use_discrete_legend = discrete_legend === nothing ? 0 < length(categories) <= legend_max_values : discrete_legend
    if use_discrete_legend
        category_colors = categorical_palette(length(categories))
        color_map = Dict(value => color for (value, color) in zip(categories, category_colors))
        mesh_colors = [value === nothing ? GLMakie.RGBAf(0, 0, 0, 1) : color_map[value] for value in vertex_values]
        plt = GLMakie.mesh!(ax, vertices, faces; color=mesh_colors, shading=false)
        elements = [GLMakie.PolyElement(color=color, strokecolor=color) for color in category_colors]
        labels = value_label.(categories)
        legend = add_value_legend!(fig, elements, labels, legend_title)
        legend === nothing || push!(decorations, legend)
        return (plot=plt, decorations=decorations)
    end

    continuous_values = continuous_vertex_values(vertex_values, categories)
    colorrange = numeric_continuous_color_range(VT, vertex_values; plot_log_transform=plot_log_transform)
    plt = GLMakie.mesh!(ax, vertices, faces; color=continuous_values, colormap=colormap, colorrange=colorrange, nan_color=:black, shading=false)
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

"""
    visualize(VT::ValuedTriangulation; kwargs...) -> GLMakie.Figure

Render a `ValuedTriangulation` using GLMakie.

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
- `legend_max_values`: categorical legend threshold, default `20`.
- `discrete_legend`: force or disable categorical legend behavior.
"""
function visualize(VT::ValuedTriangulation; kwargs...)::GLMakie.Figure
    refine_button = get(kwargs, :refine_button, true)
    button_refinement_passes = get(kwargs, :button_refinement_passes, 1)

    figure_size = get(kwargs, :figure_size, refine_button ? (1300, 900) : (900, 900))
    fig = GLMakie.Figure(size=figure_size)
    ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=GLMakie.DataAspect(), backgroundcolor=:black)
    GLMakie.colsize!(fig.layout, 1, GLMakie.Relative(1))
    GLMakie.rowsize!(fig.layout, 1, GLMakie.Relative(1))
    GLMakie.xlims!(ax, VT.xlims[1], VT.xlims[2])
    GLMakie.ylims!(ax, VT.ylims[1], VT.ylims[2])

    drawn_ref = Ref{Any}(draw_triangulation!(fig, ax, VT; kwargs...))

    refine_button && add_refine_button!(fig, ax, VT, drawn_ref; button_refinement_passes=button_refinement_passes, kwargs...)

    return fig
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

"""
    visualize(function_oracle::Function; kwargs...) -> (ValuedTriangulation, GLMakie.Figure)

Construct, refine, display, and return a `ValuedTriangulation` and its Makie
figure. `total_resolution` is interpreted as the total oracle-call budget,
including the initial mesh.
"""
function visualize(function_oracle::Function; total_resolution=nothing, max_refinement_area=nothing, verbose=true, kwargs...)
    total_given = total_resolution !== nothing
    resolved_total = total_given ? Int(total_resolution) : VALUED_TRIANGULATION_DEFAULT_TOTAL_RESOLUTION
    VT = ValuedTriangulation(function_oracle; total_resolution=resolved_total, verbose=verbose, kwargs...)
    if max_refinement_area !== nothing
        total_given && println("Warning: max_refinement_area was supplied, so total_resolution is ignored after the initial mesh.")
        refine_to_max_area!(VT, Float64(max_refinement_area); verbose=verbose)
        VT.oracle_budget = nothing
    else
        VT.oracle_budget = max(0, resolved_total - VT.total_oracle_calls)
        refine_until_budget_exhausted!(VT; verbose=verbose)
        VT.oracle_budget = nothing
    end
    fig = visualize(VT; kwargs...)
    display(fig)
    return VT, fig
end
