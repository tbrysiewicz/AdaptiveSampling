#==============================================================================#
#  ADAPTIVE VISUALIZATION
#==============================================================================#

module AdaptiveVisualization

using DelaunayTriangulation: Triangulation, triangulate, each_solid_triangle, triangle_vertices, add_point!, get_point, num_points
import GLMakie

export
    visualize,
    TriangulationCache,
    refine!,
    is_discrete,
    is_complete,
    complete_triangles,
    incomplete_triangles,
    save

include("Utilities.jl")
include("TCStruct.jl")
include("Initialization.jl")
include("WindowChange.jl")
include("Refinement.jl")
include("MakieInterface.jl")

end # module AdaptiveVisualization
