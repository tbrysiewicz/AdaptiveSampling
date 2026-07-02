include("test/HCtests.jl")


K = KuramotoModel(6)
F = restrict(K,[[1,-1,1.0,-1,1],[0,0,0,0,0.0],[1,1,1,1,1]])
F = restrict(K,[[1,-1,0.0,-0,0],[0,0,0,0,0.0],[0,0,0,1,-1]]) #beautiful symmetric picture
F = restrict(K,[[0,0,0,0,0.0],[1/3,-1/3,0.0,-0,0],[0,0,0,1/3,-1/3]]) #beautiful symmetric picture
F = restrict(K,[[0,0,0,0,0.0],[-1/3,-1/3,0.0,1/3,1/3],[1/3,1/3,0,1/3,1/3]]) #beautiful symmetric picture
f = real_solution_function(F)
 TC = TriangulationCache(f;
               xlims=[-1.5, 1.5],
               ylims=[-1.5, 1.5],
               strategy=:sierpinski,
               verbose=false,
               resolution=144,
               
               )