module AdaptiveSampling

using DelaunayTriangulation: triangulate, each_solid_triangle, triangle_vertices, get_point
import GLMakie

const FunctionCache = Vector{Tuple{Vector{Float64},Any}}
const Polygon = Vector{Int64}
const PolygonList = Vector{Polygon}
const DEFAULT_DISCRETE_LEGEND_LIMIT = 10

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

function oracle_dimension(function_oracle::Function)
    return length(methods(function_oracle)[1].sig.parameters) - 1
end

function checked_batch_oracle(function_oracle::Function)
    return function checked_oracle(points)
        values = function_oracle(points)
        if !(values isa AbstractVector) || length(values) != length(points)
            error("Batched function oracle must return one output value for each input point.")
        end
        return collect(values)
    end
end

function single_point_oracle(function_oracle::Function)
    return points -> map(p -> function_oracle(p...), points)
end

function try_evaluate_oracle(function_oracle::Function, points)
    try
        oracle = checked_batch_oracle(function_oracle)
        return oracle, oracle(points)
    catch batch_error
        try
            oracle = single_point_oracle(function_oracle)
            return oracle, oracle(points)
        catch single_point_error
            return nothing
        end
    end
end

#==============================================================================#
# EXPORTS
#==============================================================================#

export
    visualize,                   # Main user function for plotting
    visualize_function_cache,    # Plot from a function cache
    ValuedSubdivision,           # Main struct for subdivision/visualization
    refine!,                     # Refine the subdivision
    delaunay_retriangulate!,     # Re-triangulate the mesh
    is_discrete,                 # Heuristic check if function cache represents a discrete valued function
    is_complete,                 # Check if a polygon is complete
    VISUALIZATION_STRATEGIES,    # Dictionary of visualization strategies
    complete_polygons,
    incomplete_polygons,
    animate_refinement,         # Animate the refinement process
    save,
    adjust!,
    visualize_with_makie

#==============================================================================#
# MESH FUNCTIONS
#==============================================================================#

function initial_parameter_distribution(; kwargs...)
    xlims = get(kwargs, :xlims, [-1,1])
    ylims = get(kwargs, :ylims, [-1,1])
    resolution = get(kwargs, :resolution, 1000)
    xlength = abs(xlims[2] - xlims[1])
    ylength = abs(ylims[2] - ylims[1])

    nx = max(2, floor(Int, sqrt(resolution * xlength / ylength)))
    ny = max(2, floor(Int, sqrt(resolution * ylength / xlength)))

    x_values = range(xlims[1], xlims[2], length=nx)
    y_values = range(ylims[1], ylims[2], length=ny)

    return x_values, y_values
end

function trihexagonal_mesh(; kwargs...)
    xlims = get(kwargs, :xlims, [-1,1])
    ylims = get(kwargs, :ylims, [-1,1])
    resolution = get(kwargs, :resolution, 1000)

    x_values, y_values = initial_parameter_distribution(; xlims=xlims, ylims=ylims, resolution=resolution)

    shift_amount = (x_values[2] - x_values[1]) / 2
    parameters = [[x - (isodd(j_index) ? shift_amount : 0.0), y] for x in x_values for (j_index, y) in enumerate(y_values)]
    parameter_indices = Dict(point_key(p) => i for (i, p) in pairs(parameters))

    vertices = hcat(parameters...)
    tri = triangulate(vertices)
    triangles = Vector{Vector{Int}}([])
    for T in each_solid_triangle(tri)
        i, j, k = triangle_vertices(T)
        p1, p2, p3 = get_point(tri, i, j, k)
        v1 = [p1[1], p1[2]]
        v2 = [p2[1], p2[2]]
        v3 = [p3[1], p3[2]]
        idx1 = parameter_indices[point_key(v1)]
        idx2 = parameter_indices[point_key(v2)]
        idx3 = parameter_indices[point_key(v3)]
        push!(triangles, [idx1, idx2, idx3])
    end

    return (triangles, parameters)
end

function rectangular_mesh(; kwargs...)
    xlims = get(kwargs, :xlims, [-1,1])
    ylims = get(kwargs, :ylims, [-1,1])
    resolution = get(kwargs, :resolution, 1000)
    x_values, y_values = initial_parameter_distribution(; xlims=xlims, ylims=ylims, resolution=resolution)
    parameters = [[i,j] for i in x_values for j in y_values]

    nx = length(x_values)
    ny = length(y_values)
    linear_index(i, j) = (i - 1) * ny + j

    rectangles = Vector{Vector{Int}}([])
    for i in 1:(nx - 1)
        for j in 1:(ny - 1)
            tl_index = linear_index(i, j)
            tr_index = linear_index(i + 1, j)
            bl_index = linear_index(i, j + 1)
            br_index = linear_index(i + 1, j + 1)
            push!(rectangles, [tl_index, tr_index, br_index, bl_index])
        end
    end
    return (rectangles, parameters)
end

function one_dimensional_mesh(; kwargs...)
    xlims = get(kwargs, :xlims, [-1,1])
    resolution = get(kwargs, :resolution, 1000)
    parameters = range(xlims[1], xlims[2], length=resolution)
    parameters = [[p] for p in parameters]
    segments = [[i, i + 1] for i in 1:(length(parameters)-1)]
    return (segments, parameters)
end

#==============================================================================#
# LOCAL INSERTION FUNCTIONS
#==============================================================================#

function quadtree_insertion(P::Vector{Vector{Float64}})
    center = midpoint(P)
    midpoints = [midpoint([P[i], P[mod1(i+1, 4)]]) for i in 1:4]
    rectangles = [[P[i], midpoints[i], center, midpoints[mod1(i-1, 4)]] for i in 1:4]
    return (midpoints..., center), rectangles
end

function random_point_insertion(P::Vector{Vector{Float64}})
    c = abs.(randn(Float64, length(P)))
    c ./= sum(c)
    random_point = sum(P[i] .* c[i] for i in eachindex(P))
    triangles = [[P[1], P[2], random_point],
                 [P[2], P[3], random_point],
                 [P[3], P[1], random_point]]
    return [random_point], triangles
end

function sierpinski_point_insertion(P::Vector{Vector{Float64}})
    m1 = midpoint([P[1], P[2]])
    m2 = midpoint([P[1], P[3]])
    m3 = midpoint([P[2], P[3]])
    triangles = [[P[1], m1, m2],
                 [P[2], m3, m1],
                 [P[3], m2, m3],
                 [m1, m2, m3]]
    return [m1, m2, m3], triangles
end

function barycenter_point_insertion(P::Vector{Vector{Float64}})
    barycenter = midpoint(P)
    triangles = [[P[1], P[2], barycenter],
                 [P[2], P[3], barycenter],
                 [P[3], P[1], barycenter]]
    return [barycenter], triangles
end

function one_dimensional_midpoint_insertion(P::Vector{Vector{Float64}})
    midpoint_value = midpoint(P)
    segments = [[P[1], midpoint_value], [midpoint_value, P[2]]]
    return [midpoint_value], segments
end

#==============================================================================#
# VISUALIZATION STRATEGIES
#==============================================================================#

"""
    VISUALIZATION_STRATEGIES

A dictionary of visualization strategies, each containing a mesh function and a refinement method.
"""
global VISUALIZATION_STRATEGIES = Dict{Symbol, Dict{Symbol, Any}}(
    :quadtree => Dict{Symbol, Any}(:mesh_function => rectangular_mesh, :refinement_method => quadtree_insertion),
    :barycentric => Dict{Symbol, Any}(:mesh_function => trihexagonal_mesh, :refinement_method => barycenter_point_insertion),
    :sierpinski => Dict{Symbol, Any}(:mesh_function => trihexagonal_mesh, :refinement_method => sierpinski_point_insertion),
    :random => Dict{Symbol, Any}(:mesh_function => trihexagonal_mesh, :refinement_method => random_point_insertion),
    :careful => Dict{Symbol, Any}(:mesh_function => trihexagonal_mesh, :refinement_method => barycenter_point_insertion),
    :onedimensional => Dict{Symbol, Any}(:mesh_function => one_dimensional_mesh, :refinement_method => one_dimensional_midpoint_insertion)
)

#==============================================================================#
# FUNCTIONS NEEDED TO INITIALIZE VALUEDSUBDIVISION
#==============================================================================#

"""
    is_discrete(function_cache::Vector{Tuple{Vector{Float64},Any}})

Heuristically check if the output values in the function cache are discrete. A function is considered discrete if it has fewer than 50 unique output values.
"""
function is_discrete(function_cache::AbstractVector{<:Tuple})
    # Check if the output values are discrete
    output_values = getindex.(function_cache, 2)
    unique_count = length(unique(output_values))
    return unique_count < 50
end

"""
    is_complete(p::Vector{Int}, FC::Vector{Tuple{Vector{Float64},Any}}; tol=0.0, kwargs...) 

    The default function to determine if a polygon is complete.
    A polygon is considered complete if all its vertices have been evaluated and the output values are within a specified tolerance.
"""
function is_complete(polygon::Vector{Int}, FC::AbstractVector{<:Tuple}; tol = 0.0)
    vals = [FC[v][2] for v in polygon]
    vertex_function_values = sort(filter(is_real_value, vals))
    isempty(vertex_function_values) && return false
    return (vertex_function_values[end] - vertex_function_values[1]) <= tol
end

function default_is_complete(function_cache::AbstractVector{<:Tuple})
    # Checking whether the function is continuous or discrete

    is_disc = is_discrete(function_cache)
    tol = 0.0
    if !is_disc
        values = filter(is_real_value, getindex.(function_cache, 2))
        if !isempty(values)
            tol = (maximum(values) - minimum(values))/16
        end
    end
    return (p::Vector{Int}, FC::AbstractVector{<:Tuple}; kwargs...) -> is_complete(p, FC; tol=tol, kwargs...)
end

#==============================================================================#
# VALUEDSUBDIVISION STRUCTURE
#==============================================================================#

"""
    ValuedSubdivision

    A mutable struct representing a subdivision of a function's domain into polygons and a cache of function values over the vertices of
        those polygons.
    The fields of a ValuedSubdivision include:
        - `function_oracle`: A batched evaluator that takes a vector of input points and returns one output value for each point.
        - `function_cache`: A vector of tuples, each containing one sampled input point and the corresponding output value.
        - `complete_polygons`: A vector of vectors, each containing indices of the function_cache that represent complete polygons.
        - `incomplete_polygons`: A vector of vectors, each containing indices of the function_cache that represent incomplete polygons.
        - `is_complete`: A function that checks if a polygon is complete,
    
    To construct a ValuedSubdivision, provide either a batched oracle `f(points)` that returns one value for each point, or a single-point oracle
    `f(x)` / `f(x, y)`. Single-point oracles are wrapped into batched evaluators internally. For a one-dimensional batched oracle,
    pass `strategy=:onedimensional` so the initial mesh has one-coordinate points. Keyword arguments include:
        - `xlims`: The limits for the x-axis (default is [-1, 1]).
        - `ylims`: The limits for the y-axis (default is [-1, 1]).
        - `resolution`: The number of points in the subdivision (default is 1000).
        - `strategy`: A preset strategy for generating the subdivision (default is :sierpinski).
        - `mesh_function`: A custom mesh function to generate the initial mesh.
    
    # Example 
    julia> f(x,y) = x + y
    julia> VSD = ValuedSubdivision(f; resolution=3000, strategy=:quadtree)
    ValuedSubdivision with 2916 function cache entries, 2277 complete polygons, and 532 incomplete polygons.
"""
mutable struct ValuedSubdivision
    function_oracle::Function                           # Batched evaluator: points -> values
    function_cache::FunctionCache
    point_indices::Dict{Tuple,Int64}
    complete_polygons::PolygonList                      # Given as indices of function_cache
    incomplete_polygons::PolygonList
    is_complete::Function
    dimension::Int64
    strategy::Symbol

    function ValuedSubdivision(function_oracle::Function; kwargs...)
        #Pull kwargs
        xlims = get(kwargs, :xlims, [-1,1])
        ylims = get(kwargs, :ylims, [-1,1])
        resolution = get(kwargs, :resolution, 1000)
        strategy = get(kwargs, :strategy, :sierpinski)
        strategy_was_provided = haskey(kwargs, :strategy)
        candidate_strategies = Symbol[strategy]
        if !strategy_was_provided && oracle_dimension(function_oracle) == 1 && strategy != :onedimensional
            push!(candidate_strategies, :onedimensional)
        end

        polygons = nothing
        parameters = nothing
        function_oracle_values = nothing
        batched_oracle = nothing
        dimension = nothing
        effective_strategy = nothing

        for candidate_strategy in candidate_strategies
            if !haskey(VISUALIZATION_STRATEGIES, candidate_strategy)
                error("Invalid strategy inputted. Valid strategies include:",keys(VISUALIZATION_STRATEGIES),".")
            end

            mesh_function = get(kwargs, :mesh_function, VISUALIZATION_STRATEGIES[candidate_strategy][:mesh_function])
            candidate_polygons, candidate_parameters = mesh_function(; xlims=xlims, ylims=ylims, resolution=resolution, kwargs...)
            evaluation = try_evaluate_oracle(function_oracle, candidate_parameters)
            evaluation === nothing && continue

            batched_oracle, function_oracle_values = evaluation
            polygons = candidate_polygons
            parameters = candidate_parameters
            dimension = length(first(parameters))
            effective_strategy = candidate_strategy
            break
        end

        batched_oracle === nothing && error("Function oracle must be either batched, f(points) -> values, or single-point, f(x) / f(x, y) -> value.")

        VSD = new()
        VSD.function_oracle = batched_oracle
        VSD.dimension = dimension
        VSD.strategy = effective_strategy

        length(function_oracle_values) != length(parameters) && error("Did not evaluate at each parameter")

        # Create function cache
        function_cache = FunctionCache()
        for i in eachindex(function_oracle_values)
            push!(function_cache, (parameters[i], function_oracle_values[i]))
        end
        VSD.function_cache = function_cache
        VSD.point_indices = point_index_dict(function_cache)
        VSD.complete_polygons = PolygonList()
        VSD.incomplete_polygons = PolygonList(polygons)

        # If is_complete was not passed, determine it. 

        if haskey(kwargs, :is_complete)
            ic = kwargs[:is_complete]
            VSD.is_complete =  (p::Vector{Int}, FC::AbstractVector{<:Tuple}; kwargs...) -> ic(p, FC; kwargs...) # If a custom is_complete function is provided, use it
        else
            VSD.is_complete = default_is_complete(function_cache)
        end
        
        check_completeness!(VSD)
        return VSD
    end
end

function Base.show(io::IO, VSD::ValuedSubdivision)
    print(io, "ValuedSubdivision with ", length(VSD.function_cache), " function cache entries, ",
          length(VSD.complete_polygons), " complete polygons, and ",
          length(VSD.incomplete_polygons), " incomplete polygons.")
end

#==============================================================================#
# GETTERS AND SETTERS
#==============================================================================#

function_oracle(VSD::ValuedSubdivision) = VSD.function_oracle
function_cache(VSD::ValuedSubdivision) = VSD.function_cache
point_indices(VSD::ValuedSubdivision) = VSD.point_indices
"""
    complete_polygons(VSD::ValuedSubdivision)

    A vector of vectors, each containing indices of the function_cache that represent complete polygons.
"""
complete_polygons(VSD::ValuedSubdivision) = VSD.complete_polygons

"""
    incomplete_polygons(VSD::ValuedSubdivision)

    A vector of vectors, each containing indices of the function_cache that represent incomplete polygons.
"""
incomplete_polygons(VSD::ValuedSubdivision) = VSD.incomplete_polygons
input_points(FC::AbstractVector{<:Tuple}) = getindex.(FC, 1)
output_values(FC::AbstractVector{<:Tuple}) = getindex.(FC, 2)

"""
    input_points(VSD::ValuedSubdivision)

    Extracts the points from a ValuedSubdivision, returning a vector of vectors of Float64.
"""
input_points(VSD::ValuedSubdivision) = input_points(function_cache(VSD))

"""
    output_values(VSD::ValuedSubdivision)

    Extracts the output values (function oracle values) from a ValuedSubdivision, returning a vector.
"""
output_values(VSD::ValuedSubdivision) = output_values(function_cache(VSD))
is_discrete(VSD::ValuedSubdivision) = is_discrete(function_cache(VSD))

function is_complete(polygon::Vector{Int}, VSD::ValuedSubdivision; kwargs...) 
    VSD.is_complete(polygon, function_cache(VSD); kwargs...)
end

dimension(VSD::ValuedSubdivision) = VSD.dimension
strategy(VSD::ValuedSubdivision) = VSD.strategy

function find_point_index(VSD::ValuedSubdivision, point::AbstractVector)
    return get(point_indices(VSD), point_key(point), nothing)
end

function point_indices_for_polygon(VSD::ValuedSubdivision, polygon_points::AbstractVector)
    polygon = Polygon()
    for point in polygon_points
        index = find_point_index(VSD, point)
        index === nothing && error("Point $point not found in function cache.")
        push!(polygon, index)
    end
    return polygon
end

function set_function_cache!(VSD::ValuedSubdivision, FC::FunctionCache)
    VSD.function_cache = FC
    VSD.point_indices = point_index_dict(FC)
    return VSD
end

function current_polygon_arity(VSD::ValuedSubdivision)
    polygon_list = !isempty(incomplete_polygons(VSD)) ? incomplete_polygons(VSD) : complete_polygons(VSD)
    isempty(polygon_list) && error("Cannot infer refinement strategy from an empty subdivision.")
    arities = unique(length.(polygon_list))
    length(arities) == 1 || error("Cannot infer refinement strategy from mixed polygon arities: $arities.")
    return first(arities)
end

function default_refinement_strategy(VSD::ValuedSubdivision)
    n = current_polygon_arity(VSD)
    n == 4 && return :quadtree
    n == 3 && return :sierpinski
    n == 2 && return :onedimensional
    error("Cannot infer refinement strategy from polygons with $n vertices.")
end

#==============================================================================#
# POLYGON MANAGEMENT
#==============================================================================#

function delete_from_incomplete_polygons!(VSD::ValuedSubdivision, P::Vector{Int64}; kwargs...)
    polygon_index = findfirst(x -> x == P, incomplete_polygons(VSD))
    polygon_index === nothing && error("Polygon not found in IncompletePolygons!")
    deleteat!(VSD.incomplete_polygons, polygon_index)
end

function delete_from_polygons!(VSD::ValuedSubdivision, P::Vector{Int64})
    polygon_index = findfirst(==(P), incomplete_polygons(VSD))
    if polygon_index !== nothing
        return deleteat!(VSD.incomplete_polygons, polygon_index)
    end
    polygon_index = findfirst(==(P), complete_polygons(VSD))
    polygon_index === nothing && error("Polygon not found in ValuedSubdivision!")
    deleteat!(VSD.complete_polygons, polygon_index)
end

function check_completeness!(VSD::ValuedSubdivision; kwargs...)
    complete_bucket = PolygonList()
    incomplete_bucket = PolygonList()
    for p in incomplete_polygons(VSD)
        push!(is_complete(p, VSD; kwargs...) ? complete_bucket : incomplete_bucket, p)
    end
    append!(VSD.complete_polygons, complete_bucket)
    VSD.incomplete_polygons = incomplete_bucket
    return VSD
end

set_complete_polygons!(VSD::ValuedSubdivision, P::Vector{Vector{Int64}}) = (VSD.complete_polygons = P)
set_incomplete_polygons!(VSD::ValuedSubdivision, P::Vector{Vector{Int64}}) = (VSD.incomplete_polygons = P)
push_to_complete_polygons!(VSD::ValuedSubdivision, P::Vector{Int64}; kwargs...) = push!(VSD.complete_polygons, P)
push_to_incomplete_polygons!(VSD::ValuedSubdivision, P::Vector{Int64}; kwargs...) = push!(VSD.incomplete_polygons, P)

refinement_polygons(VSD::ValuedSubdivision, refine_complete::Bool) =
    refine_complete ? vcat(incomplete_polygons(VSD), complete_polygons(VSD)) : incomplete_polygons(VSD)

function push_to_function_cache!(VSD::ValuedSubdivision, V::Tuple{Vector{Float64},Any}; kwargs...)
    key = point_key(V[1])
    existing_index = get(point_indices(VSD), key, nothing)
    existing_index !== nothing && return existing_index
    push!(VSD.function_cache, V)
    index = length(function_cache(VSD))
    VSD.point_indices[key] = index
    return index
end

function evaluate_and_cache!(VSD::ValuedSubdivision, parameters::Vector{Vector{Float64}})
    values = function_oracle(VSD)(parameters)
    length(values) != length(parameters) && error("Did not evaluate at each parameter")
    for i in eachindex(values)
        push_to_function_cache!(VSD, (parameters[i], values[i]))
    end
end

#==============================================================================#
# REFINEMENT FUNCTION
#==============================================================================#

function area_of_polygon(P::Vector{Vector{Float64}})::Float64
    # Computes the area of a simple polygon using the shoelace formula
    n = length(P)
    area = 0.0
    for i in 1:n
        x1, y1 = P[i]
        x2, y2 = P[mod1(i+1, n)]
        area += x1 * y2 - x2 * y1
    end
    return 0.5 * abs(area)
end

function uncached_refinement_points(VSD::ValuedSubdivision, new_params, queued_parameter_keys)
    candidate_new_params = Vector{Vector{Float64}}()
    candidate_keys = Set{Tuple}()
    for p in new_params
        key = point_key(p)
        if !haskey(point_indices(VSD), key) && !(key in queued_parameter_keys) && !(key in candidate_keys)
            push!(candidate_new_params, p)
            push!(candidate_keys, key)
        end
    end
    return candidate_new_params, candidate_keys
end

function refine_once!(VSD::ValuedSubdivision, resolution::Int64, local_refinement_method::Function; refine_complete=false)
    refined_polygons = PolygonList()
    polygons_to_evaluate_and_sort::Vector{Vector{Vector{Float64}}} = []
    new_parameters_to_evaluate::Vector{Vector{Float64}} = []
    queued_parameter_keys = Set{Tuple}()
    resolution_used = 0

    IP = dimension(VSD) == 1 ?
        sort(refinement_polygons(VSD, refine_complete), by = x -> abs((function_cache(VSD)[x[1]][1] - function_cache(VSD)[x[2]][1])[1]), rev = true) :
        sort(refinement_polygons(VSD, refine_complete), by = x -> area_of_polygon([function_cache(VSD)[v][1] for v in x]), rev = true)

    for P in IP
        P_parameters = [function_cache(VSD)[v][1] for v in P]
        new_params, new_polygons = local_refinement_method(P_parameters)
        candidate_new_params, candidate_keys = uncached_refinement_points(VSD, new_params, queued_parameter_keys)
        isempty(candidate_new_params) && continue

        append!(new_parameters_to_evaluate, candidate_new_params)
        union!(queued_parameter_keys, candidate_keys)
        resolution_used += length(candidate_new_params)
        push!(refined_polygons, P)
        push!(polygons_to_evaluate_and_sort, new_polygons...)
        resolution_used >= resolution && break
    end

    resolution_used == 0 && return 0

    for P in refined_polygons
        refine_complete ? delete_from_polygons!(VSD, P) : delete_from_incomplete_polygons!(VSD, P)
    end

    evaluate_and_cache!(VSD, new_parameters_to_evaluate)

    for P in polygons_to_evaluate_and_sort
        polygon = point_indices_for_polygon(VSD, P)
        push_to_incomplete_polygons!(VSD, polygon)
    end

    check_completeness!(VSD)
    return resolution_used
end

function add_refinement_points_once!(VSD::ValuedSubdivision, resolution::Int64, local_refinement_method::Function; refine_complete=false)
    new_parameters_to_evaluate::Vector{Vector{Float64}} = []
    queued_parameter_keys = Set{Tuple}()
    resolution_used = 0

    IP = sort(refinement_polygons(VSD, refine_complete), by = x -> area_of_polygon([function_cache(VSD)[v][1] for v in x]), rev = true)

    for P in IP
        P_parameters = [function_cache(VSD)[v][1] for v in P]
        new_params, _ = local_refinement_method(P_parameters)
        candidate_new_params, candidate_keys = uncached_refinement_points(VSD, new_params, queued_parameter_keys)
        isempty(candidate_new_params) && continue

        append!(new_parameters_to_evaluate, candidate_new_params)
        union!(queued_parameter_keys, candidate_keys)
        resolution_used += length(candidate_new_params)
        resolution_used >= resolution && break
    end

    resolution_used == 0 && return 0

    evaluate_and_cache!(VSD, new_parameters_to_evaluate)

    return resolution_used
end

function refine!(VSD::ValuedSubdivision, resolution::Int64;	strategy = nothing, kwargs...)
    if strategy === nothing
        strategy = default_refinement_strategy(VSD)
    end

    if !haskey(VISUALIZATION_STRATEGIES, strategy)
        error("Invalid strategy inputted. Valid strategies include:",keys(VISUALIZATION_STRATEGIES),".")
    end
    if dimension(VSD) == 1 && strategy != :onedimensional
        error("One-dimensional subdivisions must use the :onedimensional strategy.")
    elseif dimension(VSD) > 1 && strategy == :onedimensional
        error("Two-dimensional subdivisions cannot use the :onedimensional strategy.")
    end
    VSD.strategy = strategy

    local_refinement_method = get(kwargs, :refinement_method, VISUALIZATION_STRATEGIES[strategy][:refinement_method])
    refine_complete = get(kwargs, :refine_complete, false)

    if strategy == :careful && dimension(VSD) > 1
        delaunay_retriangulate!(VSD)
    end

    expected_arity = strategy == :quadtree ? 4 : strategy == :onedimensional ? 2 : 3
    actual_arity = current_polygon_arity(VSD)
    actual_arity == expected_arity || error("Strategy $strategy cannot refine polygons with $actual_arity vertices.")

    total_resolution_used = 0
    remaining = resolution
    if strategy == :careful && dimension(VSD) > 1
        while remaining > 0
            used_this_pass = add_refinement_points_once!(VSD, remaining, local_refinement_method; refine_complete=refine_complete)
            used_this_pass == 0 && break
            delaunay_retriangulate!(VSD)
            total_resolution_used += used_this_pass
            remaining = resolution - total_resolution_used
        end
    else
        while remaining > 0
            used_this_pass = refine_once!(VSD, remaining, local_refinement_method; refine_complete=refine_complete)
            used_this_pass == 0 && break
            total_resolution_used += used_this_pass
            remaining = resolution - total_resolution_used
        end
    end

    println("Resolution used:", total_resolution_used)
    return(VSD)
end

"""
    refine!(VSD::ValuedSubdivision; kwargs...)

    Refines a ValuedSubdivision by adding samples to incomplete polygons.
    Local strategies replace refined polygons directly; `:careful` adds the chosen samples and then rebuilds the global Delaunay triangulation.
    
    # Keyword Arguments
        - `resolution`: An upper bound on the number of new points to be added to the subdivision during refinement (default is 1000000).
        - `strategy`: The refinement strategy to be used. If omitted, the strategy is inferred from the current mesh: `:quadtree` for quadrilaterals, `:sierpinski` for triangles, and `:onedimensional` for segments.
    
    # Example
    julia> f(x,y) = x + y
    julia> VSD = ValuedSubdivision(f; resolution=3000, strategy=:quadtree)
    ValuedSubdivision with 2916 function cache entries, 2277 complete polygons, and 532 incomplete polygons.
    julia> refine!(VSD)
    Resolution used:2092
    ValuedSubdivision with 5008 function cache entries, 3493 complete polygons, and 912 incomplete polygons.
"""
function refine!(VSD::ValuedSubdivision; kwargs...)
    return refine!(VSD, 1000000; kwargs...)
end

#==============================================================================#
# DELAUNAY RETRIANGULATION
#==============================================================================#

"""
    delaunay_retriangulate!(VSD::ValuedSubdivision)

    Re-triangulates the mesh of a ValuedSubdivision using Delaunay triangulation.
    This function ensures that the triangulation does not contain duplicate points and updates the polygons accordingly.
"""
function delaunay_retriangulate!(VSD::ValuedSubdivision)
    dimension(VSD) == 2 || error("Delaunay retriangulation is only defined for 2D subdivisions.")
    println("Delaunay re-triangulating the mesh...")
    vertices = Vector{Vector{Float64}}()
    seen = Set{Tuple}()
    for v in function_cache(VSD)
        key = point_key(v[1])
        if !(key in seen)
            push!(vertices, v[1]) #ensuring that the triangulation will not contain duplicate points and the package won't give that annoying warning
            push!(seen, key)
        end
    end
    length(vertices) >= 3 || error("At least three unique points are required for Delaunay retriangulation.")
    vertices = hcat(vertices...)
    tri = triangulate(vertices) #This comes from DelaunayTriangulation.jl
    triangle_iterator = each_solid_triangle(tri) #Also from DelaunayTriangulation.jl
    triangles = PolygonList()
    for T in triangle_iterator
        i,j,k = triangle_vertices(T)
        i,j,k = get_point(tri, i, j, k)
        triangle = point_indices_for_polygon(VSD, [[i[1], i[2]], [j[1], j[2]], [k[1], k[2]]])
        push!(triangles, triangle)
    end
    set_complete_polygons!(VSD, PolygonList())
    set_incomplete_polygons!(VSD, triangles)
    check_completeness!(VSD)
    return VSD
end

#==============================================================================#
# UTILITY FUNCTIONS
#==============================================================================#

midpoint(P::AbstractVector) = isempty(P) ? nothing : sum(P) ./ length(P)

mean(P::Vector{T}) where T <: Any = midpoint(P)

function median(V::AbstractVector)
    Vsorted = sort(V)
    n = length(Vsorted)
    if n == 0
        error("Cannot compute median of empty vector")
    elseif isodd(n)
        return Vsorted[(n+1) ÷ 2]
    else
        return (Vsorted[n ÷ 2] + Vsorted[n ÷ 2 + 1]) / 2
    end
end 

#==============================================================================#
# VISUALIZATION FUNCTIONS
#==============================================================================#

function finite_color_range(values::AbstractVector{<:Real})
    lo, hi = extrema(values)
    if lo == hi
        pad = max(abs(lo), 1.0) / 2
        return (lo - pad, hi + pad)
    end
    return (lo, hi)
end

function finite_color_range_or_default(values)
    finite_values = filter(isfinite, values)
    return isempty(finite_values) ? (0.0, 1.0) : finite_color_range(finite_values)
end

function finite_unique_values(values)
    return sort(unique(filter(isfinite, Float64.(values))))
end

function value_label(value::Real)
    return value == round(value) ? string(Int(round(value))) : string(value)
end

function categorical_palette(n::Integer)
    base_colors = GLMakie.Makie.wong_colors()
    if n <= length(base_colors)
        return base_colors[1:n]
    end

    extra_colors = map(GLMakie.Makie.to_color, [:purple, :brown, :pink, :gray, :olive, :cyan])
    colors = vcat(base_colors, extra_colors)
    while length(colors) < n
        append!(colors, colors)
    end
    return colors[1:n]
end

function categorical_color_vector(values, categories, category_colors; missing_color=GLMakie.RGBAf(1, 1, 1, 1))
    color_map = Dict(value => color for (value, color) in zip(categories, category_colors))
    return [isfinite(value) ? color_map[value] : missing_color for value in values]
end

function should_use_discrete_legend(values; legend_max_values=DEFAULT_DISCRETE_LEGEND_LIMIT, discrete_legend=nothing)
    categories = finite_unique_values(values)
    use_legend = discrete_legend === nothing ? 0 < length(categories) <= legend_max_values : discrete_legend
    return use_legend, categories
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
        is_real_value(value) && push!(numeric_values, Float64(value))
    end
    isempty(numeric_values) && return nothing
    return sum(numeric_values) / length(numeric_values)
end

function polygon_plot_value(VSD::ValuedSubdivision, polygon::Vector{Int}; plot_log_transform=false)
    value = numeric_mean_or_nothing(map(last, function_cache(VSD)[polygon]))
    value === nothing && return nothing
    if plot_log_transform
        value <= -1 && return nothing
        value = log(value + 1)
    end
    return isfinite(value) ? value : nothing
end

function coordinate_limits(VSD::ValuedSubdivision, coordinate::Int)
    values = map(x -> x[1][coordinate], function_cache(VSD))
    return [minimum(values), maximum(values)]
end

function value_limits(values::AbstractVector{<:Real})
    lo, hi = extrema(values)
    if lo == hi
        pad = max(abs(lo), 1.0) / 2
        return [lo - pad, hi + pad]
    end
    pad = (hi - lo) / 20
    return [lo - pad, hi + pad]
end

function selected_polygons(VSD::ValuedSubdivision, plot_all_polygons::Bool)
    if plot_all_polygons || isempty(complete_polygons(VSD))
        return vcat(complete_polygons(VSD), incomplete_polygons(VSD))
    end
    return complete_polygons(VSD)
end

function polygon_mesh_data(VSD::ValuedSubdivision, polygon_list::PolygonList, polygon_values)
    valid_polygons = [(P, value) for (P, value) in zip(polygon_list, polygon_values) if length(P) >= 3]
    total_vertices = sum(length(first(item)) for item in valid_polygons; init=0)
    total_faces = sum(length(first(item)) - 2 for item in valid_polygons; init=0)

    vertices = Matrix{Float64}(undef, total_vertices, 2)
    faces = Matrix{Int64}(undef, total_faces, 3)
    colors = fill(NaN, total_vertices)

    vertex_offset = 0
    face_index = 1
    for (polygon, value) in valid_polygons
        color_value = value === nothing ? NaN : value
        for (local_index, vertex_index) in pairs(polygon)
            point = function_cache(VSD)[vertex_index][1]
            row = vertex_offset + local_index
            vertices[row, 1] = point[1]
            vertices[row, 2] = point[2]
            colors[row] = color_value
        end
        for local_index in 2:(length(polygon) - 1)
            faces[face_index, 1] = vertex_offset + 1
            faces[face_index, 2] = vertex_offset + local_index
            faces[face_index, 3] = vertex_offset + local_index + 1
            face_index += 1
        end
        vertex_offset += length(polygon)
    end

    return vertices, faces, colors
end

function polygon_mesh_data(VSD::ValuedSubdivision; plot_all_polygons::Bool, plot_log_transform::Bool)
    polygon_list = selected_polygons(VSD, plot_all_polygons)
    polygon_values = [polygon_plot_value(VSD, P; plot_log_transform=plot_log_transform) for P in polygon_list]
    return polygon_mesh_data(VSD, polygon_list, polygon_values)
end

"""
    visualize_function_cache(VSD::ValuedSubdivision)

    Visualizes the function cache of a ValuedSubdivision by scatter plotting the input points and their corresponding output values.
    This function is useful for quickly visualizing the cached values without needing to create a full subdivision visualization.
"""
function visualize_function_cache(VSD::ValuedSubdivision; kwargs...)
    visualize(function_cache(VSD); kwargs...)
end

function visualize(FC::AbstractVector{<:Tuple}; kwargs...)
    fig = GLMakie.Figure()
    isempty(FC) && return fig

    dim = length(first(input_points(FC)))
    values = map(x -> is_real_value(x) ? Float64(x) : NaN, output_values(FC))
    numeric_values = filter(isfinite, values)
    colormap = get(kwargs, :colormap, :thermal)
    legend_max_values = get(kwargs, :legend_max_values, DEFAULT_DISCRETE_LEGEND_LIMIT)
    discrete_legend = get(kwargs, :discrete_legend, nothing)
    legend_title = get(kwargs, :legend_title, "value")
    use_discrete_legend, categories = should_use_discrete_legend(values; legend_max_values=legend_max_values, discrete_legend=discrete_legend)

    if dim == 1
        xs = map(first, input_points(FC))
        ys = isempty(numeric_values) ? zeros(length(xs)) : values
        ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="f(x)")
        if isempty(numeric_values)
            GLMakie.scatter!(ax, xs, ys; color=:black)
        elseif use_discrete_legend
            category_colors = categorical_palette(length(categories))
            colors = categorical_color_vector(values, categories, category_colors)
            GLMakie.scatter!(ax, xs, ys; color=colors)
            elements = [GLMakie.MarkerElement(color=color, marker=:circle) for color in category_colors]
            labels = value_label.(categories)
            add_value_legend!(fig, elements, labels, legend_title)
        else
            colorrange = finite_color_range(numeric_values)
            plt = GLMakie.scatter!(ax, xs, ys; color=values, colormap=colormap, colorrange=colorrange, nan_color=:white)
            GLMakie.Colorbar(fig[1, 2], plt)
        end
    else
        points = input_points(FC)
        xs = map(p -> p[1], points)
        ys = map(p -> p[2], points)
        ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=GLMakie.DataAspect(), backgroundcolor=:black)
        if isempty(numeric_values)
            GLMakie.scatter!(ax, xs, ys; color=:white)
        elseif use_discrete_legend
            category_colors = categorical_palette(length(categories))
            colors = categorical_color_vector(values, categories, category_colors)
            GLMakie.scatter!(ax, xs, ys; color=colors)
            elements = [GLMakie.MarkerElement(color=color, marker=:circle) for color in category_colors]
            labels = value_label.(categories)
            add_value_legend!(fig, elements, labels, legend_title)
        else
            colorrange = finite_color_range(numeric_values)
            plt = GLMakie.scatter!(ax, xs, ys; color=values, colormap=colormap, colorrange=colorrange, nan_color=:white)
            GLMakie.Colorbar(fig[1, 2], plt)
        end
    end

    return fig
end

"""
    visualize(VSD::ValuedSubdivision; kwargs...)
    Visualizes the ValuedSubdivision by plotting the polygons and their associated values.
    The function accepts keyword arguments for customization, such as 
        -`xlims`, 
        -`ylims`,
        -`plot_log_transform` (default is false),
        -`plot_all_polygons` (default is false in the case that function oracle is discrete, true otherwise)
        -`legend_max_values` (default is 10),
        -`discrete_legend` (default is automatic; set false to force a colorbar)
"""
function visualize(VSD::ValuedSubdivision; kwargs...) :: GLMakie.Figure
    xl = get(kwargs, :xlims, coordinate_limits(VSD, 1))
    plot_log_transform = get(kwargs, :plot_log_transform, false)
    plot_all_polygons = get(kwargs, :plot_all_polygons, !is_discrete(VSD))
    colormap = get(kwargs, :colormap, :thermal)
    legend_max_values = get(kwargs, :legend_max_values, DEFAULT_DISCRETE_LEGEND_LIMIT)
    discrete_legend = get(kwargs, :discrete_legend, nothing)
    legend_title = get(kwargs, :legend_title, "value")

    if dimension(VSD) > 1
        yl = get(kwargs, :ylims, coordinate_limits(VSD, 2))
        fig = GLMakie.Figure()
        ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="y", aspect=GLMakie.DataAspect(), backgroundcolor=:black)
        GLMakie.xlims!(ax, xl[1], xl[2])
        GLMakie.ylims!(ax, yl[1], yl[2])

        polygon_list = selected_polygons(VSD, plot_all_polygons)
        polygon_values = [polygon_plot_value(VSD, P; plot_log_transform=plot_log_transform) for P in polygon_list]
        vertices, faces, colors = polygon_mesh_data(VSD, polygon_list, polygon_values)
        numeric_values = filter(isfinite, colors)

        if size(faces, 1) > 0
            if isempty(numeric_values)
                GLMakie.mesh!(ax, vertices, faces; color=:white, shading=false)
            else
                use_discrete_legend, categories = should_use_discrete_legend(colors; legend_max_values=legend_max_values, discrete_legend=discrete_legend)
                if use_discrete_legend
                    category_colors = categorical_palette(length(categories))
                    mesh_colors = categorical_color_vector(colors, categories, category_colors)
                    GLMakie.mesh!(ax, vertices, faces; color=mesh_colors, shading=false)
                    elements = [GLMakie.PolyElement(color=color, strokecolor=color) for color in category_colors]
                    labels = value_label.(categories)
                    add_value_legend!(fig, elements, labels, legend_title)
                else
                    colorrange = finite_color_range(numeric_values)
                    plt = GLMakie.mesh!(ax, vertices, faces; color=colors, colormap=colormap, colorrange=colorrange, nan_color=:white, shading=false)
                    GLMakie.Colorbar(fig[1, 2], plt)
                end
            end
        end

        return fig
    else
        polygon_list = selected_polygons(VSD, plot_all_polygons)
        polygon_values = [polygon_plot_value(VSD, P; plot_log_transform=plot_log_transform) for P in polygon_list]
        numeric_values = Float64[value for value in polygon_values if value !== nothing]
        yl = get(kwargs, :ylims, isempty(numeric_values) ? [0.0, 1.0] : value_limits(numeric_values))

        fig = GLMakie.Figure()
        ax = GLMakie.Axis(fig[1, 1]; xlabel="x", ylabel="f(x)")
        GLMakie.xlims!(ax, xl[1], xl[2])
        GLMakie.ylims!(ax, yl[1], yl[2])

        segment_points = GLMakie.Point2f[]
        segment_colors = Float64[]
        for (polygon, value) in zip(polygon_list, polygon_values)
            value === nothing && continue
            length(polygon) == 2 || continue
            p1 = function_cache(VSD)[polygon[1]][1]
            p2 = function_cache(VSD)[polygon[2]][1]
            push!(segment_points, GLMakie.Point2f(p1[1], value))
            push!(segment_points, GLMakie.Point2f(p2[1], value))
            push!(segment_colors, value, value)
        end

        if !isempty(segment_points)
            use_discrete_legend, categories = should_use_discrete_legend(segment_colors; legend_max_values=legend_max_values, discrete_legend=discrete_legend)
            if use_discrete_legend
                category_colors = categorical_palette(length(categories))
                colors = categorical_color_vector(segment_colors, categories, category_colors)
                GLMakie.linesegments!(ax, segment_points; color=colors, linewidth=2)
                elements = [GLMakie.LineElement(color=color, linewidth=2) for color in category_colors]
                labels = value_label.(categories)
                add_value_legend!(fig, elements, labels, legend_title)
            else
                colorrange = finite_color_range(segment_colors)
                plt = GLMakie.linesegments!(ax, segment_points; color=segment_colors, colormap=colormap, colorrange=colorrange, linewidth=2)
                GLMakie.Colorbar(fig[1, 2], plt)
            end
        end

        return fig
    end
end

"""
    animate_refinement(VSD::ValuedSubdivision; steps::Int=100, resolution::Int=100, kwargs...)

    Saves a sequence of GLMakie frames showing the refinement process of a ValuedSubdivision `VSD`.
    The function accepts the following keyword arguments:
    - `steps`: The number of refinement steps to animate (default is 100).
    - `resolution`: The resolution for each refinement step (default is 100).
    - `filename`: Base filename for frame output (default is "refinement_animation").
    - `file_extension`: Frame extension (default is "png").
    - `kwargs`: Additional keyword arguments to pass to the `visualize` function for customization.
"""
function animate_refinement(VSD::ValuedSubdivision; steps::Int=100, resolution::Int=100, filename::String="refinement_animation", file_extension::String="png", kwargs...)
    saved_files = String[]
    base, ext = splitext(filename)
    if !isempty(ext)
        filename = base
        file_extension = replace(ext, "." => "")
    end
    for i in 1:steps
        refine!(VSD, resolution)
        fig = visualize(VSD; kwargs...)
        frame_filename = "$(filename)_$(lpad(i, 4, '0')).$(file_extension)"
        GLMakie.save(frame_filename, fig)
        push!(saved_files, frame_filename)
    end
    return saved_files
end

"""
    save(fig::GLMakie.Figure, filename::String)
Saves a GLMakie figure `fig` to a file with the specified `filename`.
"""
function save(fig::GLMakie.Figure, filename::String; file_extension = "png", dpi = 300)
    #prepend "OutputFiles/" if it is not already there
    if !startswith(filename, "OutputFiles/")
        filename = "OutputFiles/" * filename
    end
    #append the file extension if it is not already there
    if !endswith(filename, file_extension)
        filename *= "." * file_extension
    end
    GLMakie.save(filename, fig; px_per_unit=dpi/150)
    println("Plot saved to $filename")
    return filename
end

visualize_with_makie(args...; kwargs...) = visualize(args...; kwargs...)

#==============================================================================#
# VSD WINDOW MODIFICATION FUNCTIONS
#==============================================================================#

#this function assumes that the VSD is 2-dimensional, I haven't figured it out for a 1-dimensional VSD yet.
function adjust!(VSD::ValuedSubdivision; kwargs...)
    dimension(VSD) == 2 || error("adjust! is currently only defined for 2D subdivisions.")
    xl = get(kwargs, :xlims, [min(map(x->x[1][1],function_cache(VSD))...),max(map(x->x[1][1],function_cache(VSD))...)])
    yl = get(kwargs, :ylims, [min(map(x->x[1][2],function_cache(VSD))...),max(map(x->x[1][2],function_cache(VSD))...)])
    resolution = get(kwargs, :resolution, 1000)
    new_xlims = get(kwargs, :new_xlims, xl)
    new_ylims = get(kwargs, :new_ylims, yl)
    if (new_xlims == xl && new_ylims == yl) || !(new_xlims isa Vector{<:Real}) || !(new_ylims isa Vector{<:Real})
        error("Must pass new xlims and/or ylims values to adjust ValuedSubdivision. xlims and ylims must be real vectors.")
    end
    FO = function_oracle(VSD)
    FC = function_cache(VSD)
    evaluate_oracle_on_points(parameters) = FO(parameters)
    new_function_cache = FunctionCache()
    cached_keys = Set{Tuple}()
    for i in FC
        if i[1][1] >= new_xlims[1] && i[1][1] <= new_xlims[2] && i[1][2] >= new_ylims[1] && i[1][2] <= new_ylims[2]
            key = point_key(i[1])
            if !(key in cached_keys)
                push!(new_function_cache, i)
                push!(cached_keys, key)
            end
        end
    end
    x_values, y_values = initial_parameter_distribution(; xlims = new_xlims, ylims = new_ylims, resolution = resolution)
    new_parameters = [[x,y] for x in x_values for y in y_values]
    parameters_to_evaluate = Vector{Vector{Float64}}()
    for p in new_parameters
        key = point_key(p)
        if !(key in cached_keys)
            push!(parameters_to_evaluate, p)
            push!(cached_keys, key)
        end
    end
    function_oracle_values = evaluate_oracle_on_points(parameters_to_evaluate)
    length(function_oracle_values) != length(parameters_to_evaluate) && error("Did not evaluate at each parameter")
    for i in eachindex(function_oracle_values)
        push!(new_function_cache, (parameters_to_evaluate[i], (function_oracle_values[i])))
    end
    set_function_cache!(VSD, new_function_cache)
    delaunay_retriangulate!(VSD)
    return VSD
end

end # module AdaptiveSampling
