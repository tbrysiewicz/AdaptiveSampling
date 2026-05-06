#==============================================================================#
# TRIANGULATION-NATIVE ADAPTIVE VISUALIZATION
#==============================================================================#

using DelaunayTriangulation: triangulate, each_solid_triangle, triangle_vertices, add_point!, num_points
import GLMakie

export
    visualize,
    ValuedTriangulation,
    refine!,
    is_discrete,
    is_complete,
    complete_polygons,
    incomplete_polygons,
    save

include("VTUtilities.jl")
include("VTStruct.jl")
include("VTInitialization.jl")
include("VTRefinement.jl")
include("VTMakieInterface.jl")
include("VTButtons.jl")
