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
                oracle = single_to_batched_oracle(function_oracle)
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

function TriangulationCache(
        function_oracle::Function;
        xlims = [-1, 1],
        ylims = [-1, 1],
        resolution = TRIANGULATION_CACHE_DEFAULT_TOTAL_RESOLUTION,
        strategy::Symbol = :sierpinski,
        min_refinement_area = TRIANGULATION_CACHE_DEFAULT_MIN_REFINEMENT_AREA,
        max_refinement_area = nothing,
        is_complete = nothing,
        verbose::Bool = true,
        kwargs...)

    isempty(kwargs) || error("Unsupported TriangulationCache keyword(s): $(join(keys(kwargs), ", ")).")
    strategy in TRIANGULATION_CACHE_STRATEGIES || error("Invalid strategy $strategy. Use one of $(TRIANGULATION_CACHE_STRATEGIES).")

    resolution = Int(resolution)
    resolution >= 4 || error("resolution must be at least 4 so the initial triangulation can be two-dimensional.")

    xlimits = Float64.(collect(xlims))
    ylimits = Float64.(collect(ylims))
    length(xlimits) == 2 || error("xlims must have two entries.")
    length(ylimits) == 2 || error("ylims must have two entries.")

    parameters = triangulation_initial_points(xlimits, ylimits, resolution)
    evaluation = try_evaluate_triangulation_oracle(function_oracle, parameters)
    first(evaluation) === nothing && error(oracle_error_message(last(evaluation)))
    batched_oracle, values = evaluation

    function_values = FunctionValues(values)
    verbose && println("Building initial Delaunay triangulation globally from ", length(parameters), " sampled points.")
    tri = triangulate(Tuple.(parameters))
    tc_is_complete = is_complete === nothing ? default_is_complete(function_values) : is_complete

    TC = TriangulationCache(
        batched_oracle,
        tri,
        function_values,
        point_index_dict(parameters),
        Set{TriangleKey}(),
        tc_is_complete,
        strategy,
        length(parameters),
        nothing,
        Float64(min_refinement_area),
        max_refinement_area === nothing ? nothing : Float64(max_refinement_area),
        xlimits,
        ylimits,
        [(xlimits[1], xlimits[2], ylimits[1], ylimits[2])],
        Dict{Bool,Vector{Any}}(),
        verbose,
    )
    recompute_incomplete_triangles!(TC)
    return TC
end

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

function triangulation_initial_points(xlims::Vector{Float64}, ylims::Vector{Float64}, resolution::Integer)
    x_values, y_values = initial_parameter_distribution(; xlims=xlims, ylims=ylims, resolution=resolution)
    return [[Float64(x), Float64(y)] for x in x_values for y in y_values]
end
