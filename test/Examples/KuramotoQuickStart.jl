include("test/HCtests.jl")
function KuramotoModel(n)
    @var w[1:(n-1)], s[1:n], c[1:n]
    equations = []
    for i in 1:(n-1)
        coupling_sum = 0
        for j in 1:n
            coupling_sum += s[i] * c[j] - s[j] * c[i]
        end
        f_1 = w[i] - (1 / n) * coupling_sum
        f_2 = c[i]^2 + s[i]^2 - 1
        f_1 = subs(f_1, [s[n], c[n]] => [0, 1])
        f_2 = subs(f_2, [s[n], c[n]] => [0, 1])
        f_1 == 0 || push!(equations, f_1)
        f_2 == 0 || push!(equations, f_2)
    end
    return System(equations; variables=[s[1:n-1]..., c[1:n-1]...], parameters=[w[1:n-1]...])
end


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