# Interactive GLMakie controls for TriangulationCache figures.

function window_width(TC::TriangulationCache)
    return TC.xlims[2] - TC.xlims[1]
end

function window_height(TC::TriangulationCache)
    return TC.ylims[2] - TC.ylims[1]
end

function set_axis_window!(ax, TC::TriangulationCache)
    GLMakie.xlims!(ax, TC.xlims[1], TC.xlims[2])
    GLMakie.ylims!(ax, TC.ylims[1], TC.ylims[2])
    return ax
end

function shift_window!(TC::TriangulationCache, dx_fraction::Real, dy_fraction::Real)
    dx = Float64(dx_fraction) * window_width(TC)
    dy = Float64(dy_fraction) * window_height(TC)
    TC.xlims .+= dx
    TC.ylims .+= dy
    return TC
end

function zoom_window!(TC::TriangulationCache, factor::Real)
    factor = Float64(factor)
    factor > 0 || error("Zoom factor must be positive.")
    xcenter = sum(TC.xlims) / 2
    ycenter = sum(TC.ylims) / 2
    half_width = window_width(TC) * factor / 2
    half_height = window_height(TC) * factor / 2
    TC.xlims .= [xcenter - half_width, xcenter + half_width]
    TC.ylims .= [ycenter - half_height, ycenter + half_height]
    return TC
end

current_window(TC::TriangulationCache) = (TC.xlims[1], TC.xlims[2], TC.ylims[1], TC.ylims[2])

function window_contains(outer::NTuple{4,Float64}, inner::NTuple{4,Float64})
    return outer[1] <= inner[1] &&
        outer[2] >= inner[2] &&
        outer[3] <= inner[3] &&
        outer[4] >= inner[4]
end

function window_is_covered(TC::TriangulationCache, window=current_window(TC))
    return any(covered -> window_contains(covered, window), TC.covered_windows)
end

function record_covered_window!(TC::TriangulationCache, window=current_window(TC))
    window_is_covered(TC, window) || push!(TC.covered_windows, window)
    return TC
end

function seed_window_mesh!(TC::TriangulationCache; resolution::Integer, verbose=is_verbose(TC))
    points = triangulation_initial_points(TC.xlims, TC.ylims, resolution)
    new_points = Vector{Vector{Float64}}()
    queued_keys = Set{PointKey}()
    for p in points
        key = point_key(p)
        if !haskey(point_indices(TC), key) && !(key in queued_keys)
            push!(new_points, p)
            push!(queued_keys, key)
        end
    end
    inserted = evaluate_and_insert_points!(TC, new_points; verbose=verbose)
    TC.total_oracle_calls += inserted
    return inserted
end

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

function redraw_triangulation!(fig, ax, TC::TriangulationCache, drawn_ref; kwargs...)
    set_axis_window!(ax, TC)
    delete_drawn_triangulation!(ax, drawn_ref[])
    drawn_ref[] = draw_triangulation!(fig, ax, TC; kwargs...)
    return drawn_ref[]
end

function navigate_and_refine!(
        fig,
        ax,
        TC::TriangulationCache,
        drawn_ref;
        shift=(0.0, 0.0),
        zoom_factor=nothing,
        seed_and_refine=true,
        navigation_initial_resolution=250,
        navigation_refinement_budget=1000,
        verbose=is_verbose(TC),
        kwargs...)

    zoom_factor === nothing || zoom_window!(TC, zoom_factor)
    shift_window!(TC, shift[1], shift[2])
    needs_new_sampling = seed_and_refine && !window_is_covered(TC)
    if needs_new_sampling
        seed_window_mesh!(TC; resolution=navigation_initial_resolution, verbose=verbose)
        refine_with_budget!(TC, navigation_refinement_budget; verbose=verbose)
        record_covered_window!(TC)
    end
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
    GLMakie.rowgap!(fig.layout, 8)
    GLMakie.rowsize!(fig.layout, 2, GLMakie.Fixed(52))
    GLMakie.rowsize!(controls, 1, GLMakie.Fixed(40))

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
            redraw_triangulation!(fig, ax, TC, drawn_ref; kwargs...)
        end
    end

    GLMakie.on(refine_button.clicks) do _
        for _ in 1:button_refinement_passes
            TC.min_refinement_area *= refine_button_min_area_factor
            refine!(TC)
        end
        redraw_triangulation!(fig, ax, TC, drawn_ref; kwargs...)
    end

    navigate(; shift=(0.0, 0.0), zoom_factor=nothing, seed_and_refine=true) = navigate_and_refine!(
        fig,
        ax,
        TC,
        drawn_ref;
        shift=shift,
        zoom_factor=zoom_factor,
        seed_and_refine=seed_and_refine,
        navigation_initial_resolution=navigation_initial_resolution,
        navigation_refinement_budget=navigation_refinement_budget,
        verbose=verbose,
        kwargs...)

    GLMakie.on(left_button.clicks) do _
        navigate(shift=(-navigation_step, 0.0))
    end
    GLMakie.on(right_button.clicks) do _
        navigate(shift=(navigation_step, 0.0))
    end
    GLMakie.on(up_button.clicks) do _
        navigate(shift=(0.0, navigation_step))
    end
    GLMakie.on(down_button.clicks) do _
        navigate(shift=(0.0, -navigation_step))
    end
    GLMakie.on(zoom_in_button.clicks) do _
        navigate(zoom_factor=1 - zoom_step, seed_and_refine=false)
    end
    GLMakie.on(zoom_out_button.clicks) do _
        navigate(zoom_factor=1 + zoom_step)
    end

    return refine_button
end
