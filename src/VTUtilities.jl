# Shared helpers for triangulation-native adaptive visualization.

const FunctionCache = Vector{Tuple{Vector{Float64},Any}}
const Polygon = Vector{Int64}
const PolygonList = Vector{Polygon}
const TriangleKey = NTuple{3,Int64}
const VALUED_TRIANGULATION_STRATEGIES = (:random, :sierpinski, :barycenter)
const VALUED_TRIANGULATION_DEFAULT_TOTAL_RESOLUTION = 1000
const VALUED_TRIANGULATION_DEFAULT_INITIAL_RESOLUTION_FRACTION = 0.25
const VALUED_TRIANGULATION_DEFAULT_MIN_REFINEMENT_AREA = 1e-4
const VALUED_TRIANGULATION_DEFAULT_LEGEND_LIMIT = 20

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

point_key(point::AbstractVector) = Tuple(Float64.(point))

function point_index_dict(function_cache::AbstractVector{<:Tuple})
    indices = Dict{Tuple,Int64}()
    for (i, (point, _)) in pairs(function_cache)
        key = point_key(point)
        haskey(indices, key) || (indices[key] = i)
    end
    return indices
end

is_real_value(value) = value isa Real
is_wildcard_value(value) = value === :wildcard
non_wildcard_values(values) = filter(!is_wildcard_value, values)

function checked_batch_oracle(function_oracle::Function)
    return function checked_oracle(points)
        values = function_oracle(points)
        if !(values isa AbstractVector) || length(values) != length(points)
            error("Batched function oracle must return one output value for each input point.")
        end
        return collect(values)
    end
end

single_point_oracle(function_oracle::Function) = points -> map(p -> function_oracle(p...), points)

function initial_parameter_distribution(; kwargs...)
    xlims = get(kwargs, :xlims, [-1, 1])
    ylims = get(kwargs, :ylims, [-1, 1])
    resolution = get(kwargs, :resolution, 1000)
    xlength = abs(xlims[2] - xlims[1])
    ylength = abs(ylims[2] - ylims[1])

    nx = max(2, floor(Int, sqrt(resolution * xlength / ylength)))
    ny = max(2, floor(Int, sqrt(resolution * ylength / xlength)))

    return range(xlims[1], xlims[2], length=nx), range(ylims[1], ylims[2], length=ny)
end

input_points(FC::AbstractVector{<:Tuple}) = getindex.(FC, 1)
output_values(FC::AbstractVector{<:Tuple}) = getindex.(FC, 2)

"""
    is_discrete(function_cache)

Heuristically decide whether cached function values are discrete. This is used
to choose a default plotting style: small finite value sets get categorical
colors, while larger or continuous-looking value sets get a colorbar.
"""
function is_discrete(function_cache::AbstractVector{<:Tuple})
    return length(unique(output_values(function_cache))) < 50
end

"""
    is_complete(polygon, function_cache; tol=0.0)

Default polygon completeness predicate. A polygon is complete when all numeric,
non-`:wildcard` vertex values differ by at most `tol`. If every vertex value is
`:wildcard`, the polygon is complete.
"""
function is_complete(polygon::Vector{Int}, FC::AbstractVector{<:Tuple}; tol = 0.0)
    vals = non_wildcard_values([FC[v][2] for v in polygon])
    isempty(vals) && return true
    vertex_function_values = sort(filter(is_real_value, vals))
    isempty(vertex_function_values) && return false
    return (vertex_function_values[end] - vertex_function_values[1]) <= tol
end

function default_is_complete(function_cache::AbstractVector{<:Tuple})
    tol = 0.0
    if !is_discrete(function_cache)
        values = filter(is_real_value, output_values(function_cache))
        isempty(values) || (tol = (maximum(values) - minimum(values)) / 16)
    end
    return (p::Vector{Int}, FC::AbstractVector{<:Tuple}; kwargs...) -> is_complete(p, FC; tol=tol, kwargs...)
end

function area_of_polygon(P::Vector{Vector{Float64}})::Float64
    area = 0.0
    n = length(P)
    for i in 1:n
        j = i == n ? 1 : i + 1
        area += P[i][1] * P[j][2] - P[j][1] * P[i][2]
    end
    return 0.5 * abs(area)
end

function print_refinement_summary(resolution_used::Int, skipped_small_polygons::Int)
    println("Resolution used:", resolution_used, ". Small polygons skipped:", skipped_small_polygons)
end

function finite_color_range(values::AbstractVector{<:Real})
    lo, hi = extrema(values)
    if lo == hi
        pad = max(abs(lo), 1.0) / 2
        return (lo - pad, hi + pad)
    end
    return (lo, hi)
end

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

function add_value_legend!(fig, elements, labels, title)
    isempty(elements) && return nothing
    return GLMakie.Legend(fig[1, 1], elements, labels, title;
        tellwidth=false, tellheight=false, halign=:right, valign=:top,
        margin=(10, 10, 10, 10))
end

function numeric_mean_or_nothing(values)
    numeric_values = Float64[]
    for value in values
        !is_wildcard_value(value) && is_real_value(value) && push!(numeric_values, Float64(value))
    end
    isempty(numeric_values) && return nothing
    return sum(numeric_values) / length(numeric_values)
end
