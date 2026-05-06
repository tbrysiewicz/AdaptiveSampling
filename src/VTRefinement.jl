# Refinement strategies and refinement loops.

function triangulation_triangle_area(VT::ValuedTriangulation, triangle::Vector{Int64})
    return area_of_polygon([function_cache(VT)[i][1] for i in triangle])
end

function bounding_box_area(VT::ValuedTriangulation)
    return (VT.xlims[2] - VT.xlims[1]) * (VT.ylims[2] - VT.ylims[1])
end

scaled_min_refinement_area(VT::ValuedTriangulation) = VT.min_refinement_area * bounding_box_area(VT)

function triangle_points(VT::ValuedTriangulation, triangle::Vector{Int64})
    return [function_cache(VT)[i][1] for i in triangle]
end

function triangle_barycenter(points)
    n = length(points)
    return [sum(p[1] for p in points) / n, sum(p[2] for p in points) / n]
end

function random_triangle_point(points)
    a, b, c = rand(), rand(), rand()
    s = a + b + c
    weights = (a / s, b / s, c / s)
    return [
        sum(weights[i] * points[i][1] for i in 1:3),
        sum(weights[i] * points[i][2] for i in 1:3),
    ]
end

function sierpinski_triangle_points(points)
    return [
        [(points[1][1] + points[2][1]) / 2, (points[1][2] + points[2][2]) / 2],
        [(points[2][1] + points[3][1]) / 2, (points[2][2] + points[3][2]) / 2],
        [(points[3][1] + points[1][1]) / 2, (points[3][2] + points[1][2]) / 2],
    ]
end

function refinement_points(VT::ValuedTriangulation, triangle::Vector{Int64})
    points = triangle_points(VT, triangle)
    if strategy(VT) == :random
        return [random_triangle_point(points)]
    elseif strategy(VT) == :sierpinski
        return sierpinski_triangle_points(points)
    elseif strategy(VT) == :barycenter
        return [triangle_barycenter(points)]
    end
    error("Invalid strategy $(strategy(VT)). Use one of $(VALUED_TRIANGULATION_STRATEGIES).")
end

function uncached_refinement_points(VT::ValuedTriangulation, points, queued_keys::Set{Tuple})
    new_points = Vector{Vector{Float64}}()
    for p in points
        key = point_key(p)
        if !haskey(point_indices(VT), key) && !(key in queued_keys)
            push!(new_points, Float64.(p))
            push!(queued_keys, key)
        end
    end
    return new_points
end

function candidate_triangles(VT::ValuedTriangulation; max_area=nothing)
    min_area = scaled_min_refinement_area(VT)
    triangle_areas = [(T, triangulation_triangle_area(VT, T)) for T in incomplete_triangles(VT)]
    selected = if max_area === nothing
        [item for item in triangle_areas if item[2] > min_area]
    else
        [item for item in triangle_areas if item[2] > max_area && item[2] > min_area]
    end
    return first.(sort(selected; by=last, rev=true))
end

function count_small_refinement_triangles(VT::ValuedTriangulation)
    min_area = scaled_min_refinement_area(VT)
    return count(T -> triangulation_triangle_area(VT, T) <= min_area, incomplete_triangles(VT))
end

function triangle_intersects_window(VT::ValuedTriangulation, triangle::Vector{Int64})
    points = [function_cache(VT)[i][1] for i in triangle]
    xs = map(p -> p[1], points)
    ys = map(p -> p[2], points)
    return maximum(xs) >= VT.xlims[1] &&
        minimum(xs) <= VT.xlims[2] &&
        maximum(ys) >= VT.ylims[1] &&
        minimum(ys) <= VT.ylims[2]
end

function collect_refinement_points(VT::ValuedTriangulation, budget::Union{Nothing,Int}; max_area=nothing)
    budget === 0 && return Vector{Vector{Float64}}()
    new_points = Vector{Vector{Float64}}()
    queued_keys = Set{Tuple}()
    for T in candidate_triangles(VT; max_area=max_area)
        triangle_intersects_window(VT, T) || continue
        candidates = uncached_refinement_points(VT, refinement_points(VT, T), queued_keys)
        isempty(candidates) && continue
        allowed = budget === nothing ? length(candidates) : min(length(candidates), budget - length(new_points))
        append!(new_points, candidates[1:allowed])
        budget !== nothing && length(new_points) >= budget && break
    end
    return new_points
end

function evaluate_and_insert_points!(VT::ValuedTriangulation, points::Vector{Vector{Float64}}; verbose=is_verbose(VT))
    isempty(points) && return 0
    verbose && println("Evaluating oracle on ", length(points), " new point(s).")
    values = function_oracle(VT)(points)
    length(values) == length(points) || error("Did not evaluate at each parameter.")
    inserted = 0
    for i in eachindex(points)
        point = points[i]
        key = point_key(point)
        haskey(point_indices(VT), key) && continue
        add_point!(triangulation(VT), point[1], point[2])
        index = num_points(triangulation(VT))
        push!(VT.function_cache, (point, values[i]))
        index == length(function_cache(VT)) || error("Function cache and Delaunay point indices are out of sync.")
        VT.point_indices[key] = index
        inserted += 1
    end
    verbose && println("Delaunay update: inserted ", inserted, " point(s) incrementally with local add_point! updates; no global retriangulation.")
    sync_triangle_completeness!(VT)
    return inserted
end

function spend_oracle_budget!(VT::ValuedTriangulation, amount::Integer)
    VT.oracle_budget === nothing || (VT.oracle_budget = max(0, VT.oracle_budget - amount))
    return VT.oracle_budget
end

function refine_one_pass!(VT::ValuedTriangulation; max_refinement_area=nothing, verbose=is_verbose(VT))
    budget = remaining_oracle_budget(VT)
    points = collect_refinement_points(VT, budget; max_area=max_refinement_area)
    isempty(points) && (print_refinement_summary(0, count_small_refinement_triangles(VT)); return 0)
    verbose && println("Refinement pass: selected ", length(points), " new point(s) from incomplete triangles.")
    inserted = evaluate_and_insert_points!(VT, points; verbose=verbose)
    VT.total_oracle_calls += inserted
    spend_oracle_budget!(VT, inserted)
    print_refinement_summary(inserted, count_small_refinement_triangles(VT))
    return inserted
end

function refine_to_min_area!(VT::ValuedTriangulation, min_refinement_area::Real; verbose=is_verbose(VT))
    VT.min_refinement_area = Float64(min_refinement_area)
    resolution_used = 0
    pass = 0
    while true
        pass += 1
        verbose && println("Iterative refinement pass ", pass, ": refining until all incomplete triangles have area <= ", scaled_min_refinement_area(VT), ".")
        inserted = refine_one_pass!(VT; verbose=verbose)
        resolution_used += inserted
        inserted == 0 && break
    end
    return resolution_used
end

"""
    refine!(VT::ValuedTriangulation; by_min_area=nothing, max_refinement_area=nothing, verbose=VT.verbose)

Refine a `ValuedTriangulation`.

By default, this performs one pass over the current incomplete triangles, adding
new points according to `VT.strategy` and skipping triangles whose area is at or
below `VT.min_refinement_area * window_area`.

If `by_min_area` is supplied, `VT.min_refinement_area` is updated and refinement
repeats until a pass adds no new points.
"""
function refine!(VT::ValuedTriangulation; by_min_area=nothing, max_refinement_area=nothing, verbose=is_verbose(VT), kwargs...)
    by_min_area === nothing || return refine_to_min_area!(VT, by_min_area; verbose=verbose)
    return refine_one_pass!(VT; max_refinement_area=max_refinement_area, verbose=verbose)
end

function refine_until_budget_exhausted!(VT::ValuedTriangulation; verbose=is_verbose(VT))
    resolution_used = 0
    pass = 0
    while remaining_oracle_budget(VT) !== 0
        budget = remaining_oracle_budget(VT)
        budget === nothing && error("refine_until_budget_exhausted! requires a finite oracle_budget.")
        pass += 1
        verbose && println("Iterative refinement pass ", pass, ": remaining oracle budget before pass is ", budget, ".")
        inserted = refine_one_pass!(VT; verbose=verbose)
        resolution_used += inserted
        inserted == 0 && break
    end
    return resolution_used
end

function refine_to_max_area!(VT::ValuedTriangulation, max_area::Real; verbose=is_verbose(VT))
    if max_area < scaled_min_refinement_area(VT)
        VT.min_refinement_area = 0.99 * Float64(max_area) / bounding_box_area(VT)
        verbose && println("Lowered min_refinement_area to ", VT.min_refinement_area, " so the scaled minimum triangle area is below max_refinement_area = ", max_area, ".")
    end
    resolution_used = 0
    pass = 0
    while any(T -> triangulation_triangle_area(VT, T) > max_area, incomplete_triangles(VT))
        pass += 1
        verbose && println("Iterative refinement pass ", pass, ": refining incomplete triangles above max_refinement_area = ", max_area, ".")
        inserted = refine_one_pass!(VT; max_refinement_area=max_area, verbose=verbose)
        resolution_used += inserted
        inserted == 0 && break
    end
    return resolution_used
end
