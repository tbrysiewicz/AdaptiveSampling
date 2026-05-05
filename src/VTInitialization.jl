# Construction and oracle initialization.

function try_evaluate_triangulation_oracle(function_oracle::Function, points)
    errors = Tuple{Symbol,Any}[]
    try
        oracle = checked_batch_oracle(function_oracle)
        return oracle, oracle(points)
    catch err
        push!(errors, (:batched_points, err))
        try
            oracle = pts -> map(p -> function_oracle(p), pts)
            return oracle, oracle(points)
        catch err
            push!(errors, (:single_point_vector, err))
            try
                oracle = single_point_oracle(function_oracle)
                return oracle, oracle(points)
            catch err
                push!(errors, (:single_point_coordinates, err))
                return nothing, errors
            end
        end
    end
end

function oracle_error_message(errors)
    message = IOBuffer()
    println(message, "Function oracle must either accept a vector of points and return a vector of values, accept one point as f(point), or accept one point as f(x, y).")
    println(message, "Tried these call forms:")
    for (form, err) in errors
        print(message, "  ", form, ": ")
        showerror(message, err)
        println(message)
    end
    return String(take!(message))
end

function default_triangulation_is_complete(function_cache::AbstractVector{<:Tuple})
    all(value -> is_real_value(value) || is_wildcard_value(value), output_values(function_cache)) && return default_is_complete(function_cache)
    return (triangle::Vector{Int}, FC::AbstractVector{<:Tuple}; kwargs...) -> begin
        values = non_wildcard_values([FC[i][2] for i in triangle])
        isempty(values) && return true
        all(==(first(values)), values)
    end
end

function ValuedTriangulation(
        function_oracle::Function;
        xlims = [-1, 1],
        ylims = [-1, 1],
        total_resolution = nothing,
        initial_resolution = nothing,
        initial_resolution_fraction = VALUED_TRIANGULATION_DEFAULT_INITIAL_RESOLUTION_FRACTION,
        strategy::Symbol = :sierpinski,
        min_refinement_area = VALUED_TRIANGULATION_DEFAULT_MIN_REFINEMENT_AREA,
        max_refinement_area = nothing,
        is_complete = nothing,
        verbose::Bool = true,
        kwargs...)

    strategy in VALUED_TRIANGULATION_STRATEGIES || error("Invalid strategy $strategy. Use one of $(VALUED_TRIANGULATION_STRATEGIES).")

    total_given = total_resolution !== nothing
    resolved_total = total_given ? Int(total_resolution) : VALUED_TRIANGULATION_DEFAULT_TOTAL_RESOLUTION
    resolved_total >= 4 || error("total_resolution must be at least 4 so the initial triangulation can be two-dimensional.")
    if max_refinement_area !== nothing && total_given
        println("Warning: max_refinement_area was supplied, so total_resolution is ignored after the initial mesh.")
    end

    fraction = Float64(initial_resolution_fraction)
    0 < fraction <= 1 || error("initial_resolution_fraction must be in the interval (0, 1].")
    resolved_initial = initial_resolution === nothing ? max(4, ceil(Int, resolved_total * fraction)) : Int(initial_resolution)
    resolved_initial = max(4, min(resolved_initial, resolved_total))

    xlimits = Float64.(collect(xlims))
    ylimits = Float64.(collect(ylims))
    length(xlimits) == 2 || error("xlims must have two entries.")
    length(ylimits) == 2 || error("ylims must have two entries.")

    parameters = triangulation_initial_points(xlimits, ylimits, resolved_initial)
    evaluation = try_evaluate_triangulation_oracle(function_oracle, parameters)
    first(evaluation) === nothing && error(oracle_error_message(last(evaluation)))
    batched_oracle, values = evaluation

    function_cache = FunctionCache()
    for i in eachindex(parameters)
        push!(function_cache, (parameters[i], values[i]))
    end
    verbose && println("Building initial Delaunay triangulation globally from ", length(parameters), " sampled points.")
    tri = triangulate(Tuple.(parameters))
    vt_is_complete = is_complete === nothing ? default_triangulation_is_complete(function_cache) : is_complete

    VT = ValuedTriangulation(
        batched_oracle,
        tri,
        function_cache,
        point_index_dict(function_cache),
        Set{TriangleKey}(),
        vt_is_complete,
        strategy,
        length(parameters),
        nothing,
        Float64(min_refinement_area),
        max_refinement_area === nothing ? nothing : Float64(max_refinement_area),
        xlimits,
        ylimits,
        [(xlimits[1], xlimits[2], ylimits[1], ylimits[2])],
        verbose,
    )
    sync_triangle_completeness!(VT)
    return VT
end

function triangulation_initial_points(xlims::Vector{Float64}, ylims::Vector{Float64}, resolution::Integer)
    x_values, y_values = initial_parameter_distribution(; xlims=xlims, ylims=ylims, resolution=resolution)
    return [[Float64(x), Float64(y)] for x in x_values for y in y_values]
end
