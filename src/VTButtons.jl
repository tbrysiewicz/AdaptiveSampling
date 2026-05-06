# Interactive GLMakie controls for ValuedTriangulation figures.

function window_width(VT::ValuedTriangulation)
    return VT.xlims[2] - VT.xlims[1]
end

function window_height(VT::ValuedTriangulation)
    return VT.ylims[2] - VT.ylims[1]
end

function set_axis_window!(ax, VT::ValuedTriangulation)
    GLMakie.xlims!(ax, VT.xlims[1], VT.xlims[2])
    GLMakie.ylims!(ax, VT.ylims[1], VT.ylims[2])
    return ax
end

function shift_window!(VT::ValuedTriangulation, dx_fraction::Real, dy_fraction::Real)
    dx = Float64(dx_fraction) * window_width(VT)
    dy = Float64(dy_fraction) * window_height(VT)
    VT.xlims .+= dx
    VT.ylims .+= dy
    return VT
end

function zoom_window!(VT::ValuedTriangulation, factor::Real)
    factor = Float64(factor)
    factor > 0 || error("Zoom factor must be positive.")
    xcenter = sum(VT.xlims) / 2
    ycenter = sum(VT.ylims) / 2
    half_width = window_width(VT) * factor / 2
    half_height = window_height(VT) * factor / 2
    VT.xlims .= [xcenter - half_width, xcenter + half_width]
    VT.ylims .= [ycenter - half_height, ycenter + half_height]
    return VT
end

current_window(VT::ValuedTriangulation) = (VT.xlims[1], VT.xlims[2], VT.ylims[1], VT.ylims[2])

function window_contains(outer::NTuple{4,Float64}, inner::NTuple{4,Float64})
    return outer[1] <= inner[1] &&
        outer[2] >= inner[2] &&
        outer[3] <= inner[3] &&
        outer[4] >= inner[4]
end

function window_is_covered(VT::ValuedTriangulation, window=current_window(VT))
    return any(covered -> window_contains(covered, window), VT.covered_windows)
end

function record_covered_window!(VT::ValuedTriangulation, window=current_window(VT))
    window_is_covered(VT, window) || push!(VT.covered_windows, window)
    return VT
end

function seed_window_mesh!(VT::ValuedTriangulation; resolution::Integer, verbose=is_verbose(VT))
    points = triangulation_initial_points(VT.xlims, VT.ylims, resolution)
    new_points = Vector{Vector{Float64}}()
    queued_keys = Set{Tuple}()
    for p in points
        key = point_key(p)
        if !haskey(point_indices(VT), key) && !(key in queued_keys)
            push!(new_points, p)
            push!(queued_keys, key)
        end
    end
    inserted = evaluate_and_insert_points!(VT, new_points; verbose=verbose)
    VT.total_oracle_calls += inserted
    return inserted
end

function refine_with_budget!(VT::ValuedTriangulation, budget::Integer; verbose=is_verbose(VT))
    old_budget = VT.oracle_budget
    VT.oracle_budget = max(0, Int(budget))
    used = refine_until_budget_exhausted!(VT; verbose=verbose)
    VT.oracle_budget = old_budget
    return used
end

function max_area_from_slider_exponent(VT::ValuedTriangulation, exponent::Real)
    return bounding_box_area(VT) * 10.0 ^ (-Int(round(exponent)))
end

function default_max_area_slider_range()
    return 2:6
end

function default_max_area_slider_start(VT::ValuedTriangulation, slider_range)
    if VT.max_refinement_area === nothing || VT.max_refinement_area <= 0
        return first(slider_range)
    end
    default_start = Int(round(-log10(VT.max_refinement_area / bounding_box_area(VT))))
    return slider_range[argmin(abs.(slider_range .- default_start))]
end

function max_area_slider_label(exponent::Real)
    return "Max area: window * 1e-" * string(Int(round(exponent)))
end

function redraw_triangulation!(fig, ax, VT::ValuedTriangulation, drawn_ref; kwargs...)
    set_axis_window!(ax, VT)
    delete_drawn_triangulation!(ax, drawn_ref[])
    drawn_ref[] = draw_triangulation!(fig, ax, VT; kwargs...)
    return drawn_ref[]
end

function navigate_and_refine!(
        fig,
        ax,
        VT::ValuedTriangulation,
        drawn_ref;
        shift=(0.0, 0.0),
        zoom_factor=nothing,
        seed_and_refine=true,
        navigation_initial_resolution=250,
        navigation_refinement_budget=1000,
        verbose=is_verbose(VT),
        kwargs...)

    zoom_factor === nothing || zoom_window!(VT, zoom_factor)
    shift_window!(VT, shift[1], shift[2])
    needs_new_sampling = seed_and_refine && !window_is_covered(VT)
    if needs_new_sampling
        seed_window_mesh!(VT; resolution=navigation_initial_resolution, verbose=verbose)
        refine_with_budget!(VT, navigation_refinement_budget; verbose=verbose)
        record_covered_window!(VT)
    end
    redraw_triangulation!(fig, ax, VT, drawn_ref; kwargs...)
    return VT
end

"""
    add_refine_button!(fig, ax, VT, drawn_ref; button_refinement_passes=1, kwargs...)

Attach fixed-height interactive controls to a Makie figure. The `Refine` button
performs `button_refinement_passes` calls to `refine!(VT)`, tightening
`VT.min_refinement_area` by `refine_button_min_area_factor` before each pass.
Arrow buttons pan the window by `navigation_step`, and zoom buttons rescale the
window. Navigation seeds a new coarse mesh in the visible window and spends
`navigation_refinement_budget` oracle calls only when the new window is not
already inside a previously covered window.
"""
function add_refine_button!(
        fig,
        ax,
        VT::ValuedTriangulation,
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
        verbose=is_verbose(VT),
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
        slider_start = max_area_slider_start === nothing ? default_max_area_slider_start(VT, slider_range) : max_area_slider_start
        max_area_slider = GLMakie.Slider(controls[1, 9]; range=slider_range, startvalue=slider_start, tellwidth=true, width=260)
        max_area_label = GLMakie.Label(
            controls[1, 8],
            GLMakie.lift(max_area_slider_label, max_area_slider.value);
            tellwidth=false,
            width=130,
        )
        max_area_button = GLMakie.Button(controls[1, 10]; label="Fully Refine", tellwidth=false, width=120, height=30)

        GLMakie.on(max_area_button.clicks) do _
            refine_to_max_area!(VT, max_area_from_slider_exponent(VT, max_area_slider.value[]); verbose=verbose)
            redraw_triangulation!(fig, ax, VT, drawn_ref; kwargs...)
        end
    end

    GLMakie.on(refine_button.clicks) do _
        for _ in 1:button_refinement_passes
            VT.min_refinement_area *= refine_button_min_area_factor
            refine!(VT)
        end
        redraw_triangulation!(fig, ax, VT, drawn_ref; kwargs...)
    end

    navigate(; shift=(0.0, 0.0), zoom_factor=nothing, seed_and_refine=true) = navigate_and_refine!(
        fig,
        ax,
        VT,
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
