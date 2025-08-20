module AdaptiveSampling

import Base: getindex, iterate

using Plots
using DelaunayTriangulation: triangulate, each_solid_triangle, triangle_vertices, get_point, convert_boundary_points_to_indices

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
    save

#==============================================================================#
# MESH FUNCTIONS
#==============================================================================#

function initial_parameter_distribution(; kwargs...)
    xlims = get(kwargs, :xlims, [-1,1])
    ylims = get(kwargs, :ylims, [-1,1])
    resolution = get(kwargs, :resolution, 1000)
    xlength = xlims[2] - xlims[1]
    ylength = ylims[2] - ylims[1]

    x_values = range(xlims[1], xlims[2], Int(floor(sqrt((((xlength)/(xlength + ylength))*resolution)/(1 - ((xlength)/(xlength + ylength)))))))
    y_values = range(ylims[1], ylims[2], Int(floor(sqrt((((ylength)/(xlength + ylength))*resolution)/(1 - ((ylength)/(xlength + ylength)))))))

    return x_values, y_values
end

function trihexagonal_mesh(; kwargs...)
    xlims = get(kwargs, :xlims, [-1,1])
    ylims = get(kwargs, :ylims, [-1,1])
    resolution = get(kwargs, :resolution, 1000)

    x_values, y_values = initial_parameter_distribution(; xlims=xlims, ylims=ylims, resolution=resolution)

    shift_amount = (x_values[2] - x_values[1])/2 	
    parameters = [[i - isodd(findfirst(x->x==j, y_values))*shift_amount, j] for i in x_values for j in y_values]

    parameters_as_matrix = reshape(parameters, length(x_values), length(y_values))

    triangles = Vector{Vector{Int}}([])
    for i in 1:length(x_values)-1
        for j in 1:length(y_values)-1
            tl_index = findfirst(x->x == parameters_as_matrix[i,j], parameters)
            tr_index = findfirst(x->x == parameters_as_matrix[i+1,j], parameters)
            bl_index = findfirst(x->x == parameters_as_matrix[i, j+1], parameters)
            br_index = findfirst(x->x == parameters_as_matrix[i+1, j+1], parameters)

            if isodd(i)
                push!(triangles,[tl_index, tr_index, bl_index])
                push!(triangles,[bl_index, tr_index, br_index])
            else
                push!(triangles,[tl_index, tr_index, br_index])
                push!(triangles,[bl_index, tl_index, br_index])
            end
        end
    end
    return(triangles,parameters)
end

function rectangular_mesh(; kwargs...)
    xlims = get(kwargs, :xlims, [-1,1])
    ylims = get(kwargs, :ylims, [-1,1])
    resolution = get(kwargs, :resolution, 1000)
    x_values, y_values = initial_parameter_distribution(; xlims=xlims, ylims=ylims, resolution=resolution)
    parameters = [[i,j] for i in x_values for j in y_values]
    parameters_as_matrix = reshape(parameters, length(x_values), length(y_values))
    rectangles = Vector{Vector{Int}}([])
    for i in 1:length(x_values)-1
        for j in 1:length(y_values)-1
            tl_index = findfirst(x->x == parameters_as_matrix[i,j], parameters)
            tr_index = findfirst(x->x == parameters_as_matrix[i+1,j], parameters)
            bl_index = findfirst(x->x == parameters_as_matrix[i, j+1], parameters)
            br_index = findfirst(x->x == parameters_as_matrix[i+1, j+1], parameters)
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
function is_discrete(function_cache::Vector{Tuple{Vector{Float64},Any}})
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
function is_complete(polygon::Vector{Int}, FC::Vector{Tuple{Vector{Float64},Any}}; tol = 0.0) 
    vals = [FC[v][2] for v in polygon]
    vertex_function_values = sort(filter(x->isa(x,Number),vals))
    if length(vertex_function_values) == 0
        return false # If there are no vertices, we consider it incomplete
    end
    if (vertex_function_values[end] - vertex_function_values[1]) <= tol
        return true
    else
        return false
    end
end

function default_is_complete(function_cache::Vector{Tuple{Vector{Float64},Any}})
    # Checking whether the function is continuous or discrete

    is_disc = is_discrete(function_cache)
    tol = 0.0
    if !is_disc
        values = filter(x -> isa(x, Number), getindex.(function_cache, 2))
        tol = (max(values...) - min(values...))/16
    end
    local_is_complete = (p::Vector{Int}, FC:: Vector{Tuple{Vector{Float64},Any}};kwargs...) -> is_complete(p, FC; tol=tol, kwargs...) # Default function to determine completeness of polygons
    return local_is_complete
end

#==============================================================================#
# VALUEDSUBDIVISION STRUCTURE
#==============================================================================#

"""
    ValuedSubdivision

    A mutable struct representing a subdivision of a function's domain into polygons and a caching of the function values over the vertices of 
        those polygons. 
    The fields of a ValuedSubdivision include:
        - `function_oracle`: A function that takes many parameters and returns a vector of values.
        - `function_cache`: A vector of tuples, each containing a vector of input parameters and their corresponding output value.
        - `complete_polygons`: A vector of vectors, each containing indices of the function_cache that represent complete polygons.
        - `incomplete_polygons`: A vector of vectors, each containing indices of the function_cache that represent incomplete polygons.
        - `is_complete`: A function that checks if a polygon is complete,
    
    To construct a ValuedSubdivision, you need to provide a function that takes a pair of real numbers as input (a single parameter) and returns a value. Keyword arguments include:
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
    function_oracle::Function                           #This takes MANY parameters and returns a vector of values
    function_cache::Vector{Tuple{Vector{Float64},Any}}
    complete_polygons::Vector{Vector{Int64}}            # Given as indices of function_cache
    incomplete_polygons::Vector{Vector{Int64}}
    is_complete::Function

    function ValuedSubdivision(function_oracle::Function; kwargs...)
        #Pull kwargs
        xlims = get(kwargs, :xlims, [-1,1])
        ylims = get(kwargs, :ylims, [-1,1])
        resolution = get(kwargs, :resolution, 1000)
        strategy = get(kwargs, :strategy, :sierpinski)
        nargs = length(methods(function_oracle)[1].sig.parameters) - 1 #checking how many arguments the function_oracle takes (1 or 2)
        if nargs == 1
            strategy = :onedimensional # If the function oracle takes only one parameter, we use the one-dimensional strategy
        elseif nargs > 2
            error("Function oracle must take either 1 or 2 parameters.")
        end

        mesh_function = get(kwargs, :mesh_function, VISUALIZATION_STRATEGIES[strategy][:mesh_function])

        VSD = new()
        function_oracle_on_many_parameters(parameters) = map(p -> function_oracle(p...), parameters)
        VSD.function_oracle = function_oracle

        # Generate initial mesh
        (polygons, parameters) = mesh_function(; xlims=xlims, ylims=ylims, resolution=resolution, kwargs...)

        # Evaluate function oracle for all points in the initial mesh
        function_oracle_values = function_oracle_on_many_parameters(parameters)
        length(function_oracle_values) != length(parameters) && error("Did not evaluate at each parameter")

        # Create function cache
        function_cache = Vector{Tuple{Vector{Float64},Any}}([])
        for i in eachindex(function_oracle_values)
            push!(function_cache, (parameters[i], function_oracle_values[i]))
        end
        VSD.function_cache = function_cache
        VSD.complete_polygons = Vector{Vector{Int64}}([])
        VSD.incomplete_polygons = Vector{Vector{Int64}}(polygons)

        # If is_complete was not passed, determine it. 

        if haskey(kwargs, :is_complete)
            ic = kwargs[:is_complete]
            VSD.is_complete =  (p::Vector{Int}, FC:: Vector{Tuple{Vector{Float64},Any}};kwargs...) -> ic(p, FC;kwargs...) # If a custom is_complete function is provided, use it
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
input_points(FC::Vector{Tuple{Vector{Float64}, Any}}) = getindex.(FC, 1)
output_values(FC::Vector{Tuple{Vector{Float64}, Any}}) = getindex.(FC, 2)

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

dimension(VSD::ValuedSubdivision) = length(methods(function_oracle(VSD))[1].sig.parameters) - 1

#==============================================================================#
# POLYGON MANAGEMENT
#==============================================================================#

function delete_from_incomplete_polygons!(VSD::ValuedSubdivision, P::Vector{Int64}; kwargs...)
    polygon_index = findfirst(x -> x == P, incomplete_polygons(VSD))
    if polygon_index !== nothing
        deleteat!(VSD.incomplete_polygons, polygon_index)
    else
        error("Polygon not found in IncompletePolygons!")
    end
end

function push_to_complete_polygons!(VSD::ValuedSubdivision, P::Vector{Int64}; kwargs...)
    push!(VSD.complete_polygons, P)
end

function check_completeness!(VSD::ValuedSubdivision; kwargs...)
    # Check if the current polygons are complete
    complete_bucket = []
    for p in incomplete_polygons(VSD)
        if is_complete(p, VSD; kwargs...)
            push!(complete_bucket, p)
        end
    end
    for p in complete_bucket
        delete_from_incomplete_polygons!(VSD, p)
        push_to_complete_polygons!(VSD, p)
    end
end

function set_complete_polygons!(VSD::ValuedSubdivision, P::Vector{Vector{Int64}})
    VSD.complete_polygons = P
end

function set_incomplete_polygons!(VSD::ValuedSubdivision, P::Vector{Vector{Int64}})
    VSD.incomplete_polygons = P
end

function push_to_incomplete_polygons!(VSD::ValuedSubdivision, P::Vector{Int64}; kwargs...)
    push!(VSD.incomplete_polygons, P)
end

function push_to_function_cache!(VSD::ValuedSubdivision, V::Tuple{Vector{Float64},Any}; kwargs...)
    push!(VSD.function_cache, V)
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

function refine!(VSD::ValuedSubdivision, resolution::Int64;	strategy = nothing, kwargs...)

    if strategy === nothing
        n = length(complete_polygons(VSD)[1])
        strategy = n == 4 ? :quadtree : n == 3 ? :sierpinski : n == 2 ? :onedimensional : strategy
    end

    if in(strategy, collect(keys(VISUALIZATION_STRATEGIES))) == false
        error("Invalid strategy inputted. Valid strategies include:",keys(VISUALIZATION_STRATEGIES),".")
    end

    local_refinement_method = get(kwargs, :refinement_method, VISUALIZATION_STRATEGIES[strategy][:refinement_method])

    FO(parameters) = map(p -> function_oracle(VSD)(p...), parameters)
    refined_polygons::Vector{Vector{Int64}} = []
    polygons_to_evaluate_and_sort::Vector{Vector{Vector{Float64}}} = []
    new_parameters_to_evaluate::Vector{Vector{Float64}} = []
    resolution_used = 0

    IP = []
    if dimension(VSD) == 1
        IP = sort(incomplete_polygons(VSD), by = x -> abs((function_cache(VSD)[x[1]][1] - function_cache(VSD)[x[2]][1])[1]), rev = true)
    else
        IP = sort(incomplete_polygons(VSD), by = x -> area_of_polygon([function_cache(VSD)[v][1] for v in x]), rev = true)
    end
    for P in IP
        P_parameters = [function_cache(VSD)[v][1] for v in P]
        new_params, new_polygons = local_refinement_method(P_parameters)
        push!(refined_polygons, P)
        push!(polygons_to_evaluate_and_sort, new_polygons...)
        for p in new_params
            if !(p in new_parameters_to_evaluate) && !(p in input_points(VSD))
                push!(new_parameters_to_evaluate, p)
                resolution_used += 1
            end
        end
        resolution_used >= resolution && break
    end

    for P in refined_polygons
        delete_from_incomplete_polygons!(VSD, P)
    end

    length(new_parameters_to_evaluate) == 0 && return resolution_used
    function_oracle_values = FO(new_parameters_to_evaluate)
    length(function_oracle_values) != length(new_parameters_to_evaluate) && error("Did not evaluate at each parameter")
    for i in eachindex(function_oracle_values)
         push_to_function_cache!(VSD, (new_parameters_to_evaluate[i], (function_oracle_values[i])))
    end

    for P in polygons_to_evaluate_and_sort
        polygon = [findfirst(x->x[1]==y, function_cache(VSD)) for y in P]
        push_to_incomplete_polygons!(VSD, polygon) 
    end
    
    check_completeness!(VSD)
    println("Resolution used:", resolution_used)
    return(VSD)
end

"""
    refine!(VSD::ValuedSubdivision; kwargs...)

    Refines a ValuedSubdivision via incomplete polygon subdivision.
    
    # Keyword Arguments
        - `resolution`: An upper bound on the number of new points to be added to the subdivision during refinement (default is 1000000).
        - `strategy`: The refinement strategy to be used (default is :sierpinski in the case that the subdivision is triangular and :quadtree in the case that the subdivision is rectangular).
    
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
    println("Delaunay re-triangulating the mesh...")
    vertices = []
    for v in function_cache(VSD)
        v[1] in vertices || push!(vertices, v[1]) #ensuring that the triangulation will not contain duplicate points and the package won't give that annoying warning
    end
    vertices = hcat(vertices...)
    tri = triangulate(vertices) #This comes from DelaunayTriangulation.jl
    triangle_iterator = each_solid_triangle(tri) #Also from DelaunayTriangulation.jl
    triangles::Vector{Vector{Int64}} = []
    for T in triangle_iterator
        i,j,k = triangle_vertices(T)
        i,j,k = get_point(tri, i, j, k)
        vertex_1 = [i[1], i[2]]
        vertex_2 = [j[1], j[2]]
        vertex_3 = [k[1], k[2]]
        index_1 = findfirst(x->x[1] == vertex_1, function_cache(VSD))
        index_2 = findfirst(x->x[1] == vertex_2, function_cache(VSD))
        index_3 = findfirst(x->x[1] == vertex_3, function_cache(VSD))
        triangle = [index_1,index_2,index_3]
        push!(triangles, triangle)
    end
    set_complete_polygons!(VSD, Vector{Vector{Int64}}([])) 
    set_incomplete_polygons!(VSD, triangles)
    check_completeness!(VSD)
    return VSD
end

#==============================================================================#
# UTILITY FUNCTIONS
#==============================================================================#

function midpoint(P::AbstractVector) 
    if length(P) == 0
        return nothing
    end
    return((sum(P))./length(P))
end

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

"""
    visualize_function_cache(VSD::ValuedSubdivision)

    Visualizes the function cache of a ValuedSubdivision by scatter plotting the input points and their corresponding output values.
    This function is useful for quickly visualizing the cached values without needing to create a full subdivision visualization.
"""
function visualize_function_cache(VSD::ValuedSubdivision)
    visualize(function_cache(VSD))
end
function visualize(FC::Vector{Tuple{Vector{Float64},Any}})
    scatter(first.(input_points(FC)), last.(input_points(FC)), 
    zcolor = map(x->isa(x,Number) ? x : -10.0, output_values(FC)), legend = false, colorbar = true)
end

"""
    visualize(VSD::ValuedSubdivision; kwargs...)
    Visualizes the ValuedSubdivision by plotting the polygons and their associated values.
    The function accepts keyword arguments for customization, such as 
        -`xlims`, 
        -`ylims`,
        -`plot_log_transform` (default is false),
        -`plot_all_polygons` (default is false in the case that function oracle is discrete, true otherwise)
"""
function visualize(VSD::ValuedSubdivision; kwargs...) :: Plots.Plot
    xl = get(kwargs, :xlims, [min(map(x->x[1][1],function_cache(VSD))...),max(map(x->x[1][1],function_cache(VSD))...)])
    yl = []
    MyPlot = []
    plot_log_transform = get(kwargs, :plot_log_transform, false)
    plot_all_polygons = get(kwargs, :plot_all_polygons, is_discrete(VSD) == false)

    if dimension(VSD) > 1
        yl = get(kwargs, :ylims, [min(map(x->x[1][2],function_cache(VSD))...),max(map(x->x[1][2],function_cache(VSD))...)])
        MyPlot = plot(xlims = xl, ylims = yl, aspect_ratio = :equal, background_color_inside=:black; kwargs...)
    else
        sorted_values = sort(output_values(VSD))
        yl = [sorted_values[1] - 1, sorted_values[end] + 1]
        MyPlot = plot(xlims = xl, ylims = yl, background_color_inside = :white; kwargs...)
    end
    
    polygon_list = plot_all_polygons == true ? vcat(complete_polygons(VSD), incomplete_polygons(VSD)) : complete_polygons(VSD)
    polygons_to_draw = map(P->map(first,function_cache(VSD)[P]), polygon_list)
    # we remove non-numbers when computing means so that non-numbers function as wild cards essentially 
    polygon_values = map(P-> mean(filter(x->isa(x,Number),map(last,function_cache(VSD)[P]))), polygon_list)
    if plot_log_transform
        # If we are plotting log transformed values, we need to transform the polygon values
        polygon_values = map(x->x==nothing ? nothing : log(x+1), polygon_values)
    end
    # It is still possible that the mean returns 'nothing' if all values are non-numbers
    real_values = filter(x->x!=nothing, unique(polygon_values))
    max_value = max(real_values...)
    shapes = Shape[]
    colors = []
    
    if dimension(VSD) > 1
        for (poly_pts, value) in zip(polygons_to_draw, polygon_values)
            push!(shapes, Shape([(t[1], t[2]) for t in poly_pts]))
            push!(colors, value==nothing ? :white : cgrad(:thermal, rev=false)[value/max_value])
        end
        MyPlot = plot!(shapes; fillcolor=permutedims(colors), linecolor=permutedims(colors), linewidth=0, label = false)
        legend_values = sort(real_values, rev=true)
        if in(nothing,polygon_values)
            pushfirst!(legend_values, nothing)
        end
        if length(legend_values) <= 10
            # Add legend entries for unique values
            for val in legend_values
                color = val==nothing ? :white : cgrad(:thermal, rev=false)[val/max_value]
                # Plot an invisible shape with the correct color and label for the legend
                plot!(Shape([(NaN, NaN), (NaN, NaN), (NaN, NaN)]); fillcolor=color, linecolor=:transparent, linewidth=0, label="$val", legend = :outerright)
            end
        else
            # Add a colorbar for the continuous values
            scatter!([NaN], [NaN]; zcolor = [min(real_values...), max(real_values...)], color = cgrad(:thermal, rev = false), markersize = 0, colorbar = true, label = false)
        end
    else
        for (poly_pts, value) in zip(polygons_to_draw, polygon_values)
            plot!([poly_pts[1][1], poly_pts[2][1]], [value, value], linecolor =:white, linewidth=2, label = false)
        end
    end

    return(MyPlot)
end

"""
    animate_refinement(VSD::ValuedSubdivision; steps::Int=100, resolution::Int=100, kwargs...)

    Creates an animation of the refinement process of a ValuedSubdivision `VSD`.
    The animation shows the subdivision being refined step by step, with each step visualized using the `visualize` function.
    The function accepts the following keyword arguments:
    - `steps`: The number of refinement steps to animate (default is 100).
    - `resolution`: The resolution for each refinement step (default is 100).
    - `kwargs`: Additional keyword arguments to pass to the `visualize` function for customization
"""
function animate_refinement(VSD::ValuedSubdivision; steps::Int=100, resolution::Int=100, kwargs...)

    anim = @animate for i in 1:steps
        refine!(VSD, resolution)
        visualize(VSD; kwargs...)
    end

    gif(anim, "refinement_animation.gif", fps=10)
end

"""
    save(P::Plot, filename::String)
Saves a plot `P` to a file with the specified `filename`.
"""
function save(P::Plots.Plot, filename::String; file_extension = "png", dpi = 300)
    #prepend "OutputFiles/" if it is not already there
    if !startswith(filename, "OutputFiles/")
        filename = "OutputFiles/" * filename
    end
    #append the file extension if it is not already there
    if !endswith(filename, file_extension)
        filename *= "." * file_extension
    end
    savefig(P, filename)
    println("Plot saved to $filename")
end

end # module AdaptiveSampling
