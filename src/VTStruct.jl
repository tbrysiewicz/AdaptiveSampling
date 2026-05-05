# Core ValuedTriangulation data model and accessors.

"""
    ValuedTriangulation(function_oracle; kwargs...)

Create a triangulation-native adaptive sampling object over a rectangular
window. The Delaunay triangulation stores the mesh, `function_cache` stores
sampled point/value pairs, and `incomplete_triangles` stores only the triangles
still needing refinement.

`function_oracle` may be either batched, `f(points) -> values`, or single-point,
`f(point)` / `f(x, y)`. Batched oracles are preferred and are preserved when
detected.

Keyword arguments:
- `xlims`, `ylims`: domain limits, default `[-1, 1]`.
- `total_resolution`: total oracle-call budget used by `visualize(f; ...)`.
- `initial_resolution`: explicit initial mesh oracle-call budget.
- `initial_resolution_fraction`: initial budget as a fraction of `total_resolution`.
- `strategy`: one of `:random`, `:sierpinski`, `:barycenter`.
- `min_refinement_area`: normalized minimum triangle area; scaled by window area.
- `max_refinement_area`: optional iterative refinement target.
- `is_complete`: custom completeness predicate `(triangle, function_cache; kwargs...)`.
- `verbose`: whether to print progress.

The value `:wildcard` is special in the default completeness rule: it is treated
as equal to every other value.

`covered_windows` records rectangular viewports that have already been seeded,
so zooming back out to a previously viewed region can redraw without spending
new oracle calls.
"""
mutable struct ValuedTriangulation
    function_oracle::Function
    triangulation::Any
    function_cache::FunctionCache
    point_indices::Dict{Tuple,Int64}
    incomplete_triangles::Set{TriangleKey}
    is_complete::Function
    strategy::Symbol
    total_oracle_calls::Int64
    oracle_budget::Union{Nothing,Int64}
    min_refinement_area::Float64
    max_refinement_area::Union{Nothing,Float64}
    xlims::Vector{Float64}
    ylims::Vector{Float64}
    covered_windows::Vector{NTuple{4,Float64}}
    plot_value_order::Dict{Bool,Vector{Any}}
    verbose::Bool
end

function Base.show(io::IO, VT::ValuedTriangulation)
    print(io, "ValuedTriangulation with ", length(function_cache(VT)), " function cache entries, ",
        length(complete_triangles(VT)), " complete triangles, and ",
        length(incomplete_triangles(VT)), " incomplete triangles.")
end

function_oracle(VT::ValuedTriangulation) = VT.function_oracle
function_cache(VT::ValuedTriangulation) = VT.function_cache
triangulation(VT::ValuedTriangulation) = VT.triangulation
point_indices(VT::ValuedTriangulation) = VT.point_indices
incomplete_triangle_keys(VT::ValuedTriangulation) = VT.incomplete_triangles
incomplete_triangles(VT::ValuedTriangulation) = [collect(key) for key in incomplete_triangle_keys(VT)]

"""
    complete_polygons(VT::ValuedTriangulation)

Return complete Delaunay triangles as vectors of vertex indices. Complete
triangles are computed as the complement of the stored incomplete set.
"""
complete_polygons(VT::ValuedTriangulation) = complete_triangles(VT)

"""
    incomplete_polygons(VT::ValuedTriangulation)

Return incomplete Delaunay triangles as vectors of vertex indices.
"""
incomplete_polygons(VT::ValuedTriangulation) = incomplete_triangles(VT)

input_points(VT::ValuedTriangulation) = input_points(function_cache(VT))
output_values(VT::ValuedTriangulation) = output_values(function_cache(VT))
is_discrete(VT::ValuedTriangulation) = is_discrete(function_cache(VT))
dimension(::ValuedTriangulation) = 2
strategy(VT::ValuedTriangulation) = VT.strategy
is_verbose(VT::ValuedTriangulation) = VT.verbose
remaining_oracle_budget(VT::ValuedTriangulation) = VT.oracle_budget

function triangle_indices(T)
    i, j, k = triangle_vertices(T)
    return Int64[i, j, k]
end

canonical_triangle(triangle::Vector{Int64}) = Tuple(sort(triangle))::TriangleKey

function complete_triangles(VT::ValuedTriangulation)
    incomplete = incomplete_triangle_keys(VT)
    triangles = PolygonList()
    for T in each_solid_triangle(triangulation(VT))
        triangle = triangle_indices(T)
        canonical_triangle(triangle) in incomplete || push!(triangles, triangle)
    end
    return triangles
end

"""
    is_complete(triangle, VT::ValuedTriangulation; kwargs...)

Evaluate `VT`'s completeness predicate on a triangle represented by cached
vertex indices.
"""
function is_complete(triangle::Vector{Int64}, VT::ValuedTriangulation; kwargs...)
    return VT.is_complete(triangle, function_cache(VT); kwargs...)
end

function sync_triangle_completeness!(VT::ValuedTriangulation)
    empty!(VT.incomplete_triangles)
    for T in each_solid_triangle(triangulation(VT))
        triangle = triangle_indices(T)
        is_complete(triangle, VT) || push!(VT.incomplete_triangles, canonical_triangle(triangle))
    end
    return VT
end
