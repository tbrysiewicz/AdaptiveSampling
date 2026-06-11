# Shared helpers for adaptive visualization.

#####################################
# Type Aliases
#####################################

const FunctionValues = Vector{Any}
const TriangleList = Vector{Vector{Int64}}
const PointKey = NTuple{2,Float64}
const TriangleKey = NTuple{3,Int64}

#####################################
#  Defaults
#####################################

const TRIANGULATION_CACHE_STRATEGIES = (:random, :sierpinski, :barycenter)
const TRIANGULATION_CACHE_DEFAULT_TOTAL_RESOLUTION = 1000
const TRIANGULATION_CACHE_DEFAULT_INITIAL_RESOLUTION_FRACTION = 0.25
const TRIANGULATION_CACHE_DEFAULT_MIN_REFINEMENT_AREA = 1e-5
const TRIANGULATION_CACHE_DEFAULT_LEGEND_LIMIT = 20

###################################
# Point and Cache Helpers
###################################

# Convert a point to a cache key.
function point_key(point)::PointKey
    length(point) == 2 || error("TriangulationCache points must have dimension 2.")
    return (Float64(point[1]), Float64(point[2]))
end

# Build point-to-index lookup.
function point_index_dict(points::AbstractVector)
    indices = Dict{PointKey,Int64}()
    for (i, point) in pairs(points)
        key = point_key(point)
        haskey(indices, key) || (indices[key] = i)
    end
    return indices
end

#############################
# Oracle Adapters
#############################

# Validate a batched oracle.
function checked_batch_oracle(function_oracle::Function)
    return function checked_oracle(points)
        values = function_oracle(points)
        if !(values isa AbstractVector) || length(values) != length(points)
            error("Batched function oracle must return one output value for each input point.")
        end
        return collect(values)
    end
end

# Convert a single-point oracle.
single_to_batched_oracle(function_oracle::Function) = points -> map(p -> function_oracle(p...), points)

#############################################
# Value Classification and Completeness
#############################################

# Detect real values.
is_real_value(value) = value isa Real

# Detect wildcard values.
is_wildcard_value(value) = value === :wildcard

# Drop wildcard values.
non_wildcard_values(values) = filter(!is_wildcard_value, values)

"""
    is_discrete(function_values)

Heuristically decide whether function values are discrete. Non-real values are
always treated as discrete; numeric values use a small-cardinality heuristic.
"""
# Detect categorical/discrete values.
function is_discrete(function_values::AbstractVector)
    values = non_wildcard_values(function_values)
    #if any value is not a real value or wildcard, declare 'discrete'
    any(value -> !is_real_value(value), values) && return true
    return length(unique(values)) < 50
end

# Check value agreement.
function values_are_complete(values::AbstractVector; tol = 0.0)
    vals = non_wildcard_values(values)
    isempty(vals) && return true
    all(is_real_value, vals) || return all(==(first(vals)), vals)
    vertex_function_values = sort(vals)
    return (vertex_function_values[end] - vertex_function_values[1]) <= tol
end

# Build the default triangle predicate.
function default_is_complete(function_values::AbstractVector)
    tol = 0.0
    if !is_discrete(function_values)
        values = non_wildcard_values(function_values)
        all(is_real_value, values) || error("Non-real function values must be handled as discrete values.")
        isempty(values) || (tol = (maximum(values) - minimum(values)) / 16)
    end
    return (vertices, values::AbstractVector; kwargs...) -> values_are_complete(values; tol=tol, kwargs...)
end

# Average numeric non-wildcard values.
function numeric_mean_or_nothing(values)
    numeric_values = Float64.(filter(is_real_value, non_wildcard_values(values)))
    isempty(numeric_values) && return nothing
    return sum(numeric_values) / length(numeric_values)
end

############################
# Geometry Helpers
############################

# Compute triangle area.
area_of_triangle(P::AbstractVector)::Float64 = 0.5 * abs((P[2][1] - P[1][1]) * (P[3][2] - P[1][2]) - (P[3][1] - P[1][1]) * (P[2][2] - P[1][2]))

#############################
# Console Reporting
#############################

# Print refinement totals.
function print_refinement_summary(resolution_used::Int, skipped_small_triangles::Int)
    println("Resolution used:", resolution_used, ". Small triangles skipped:", skipped_small_triangles)
end
