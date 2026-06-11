
using AdaptiveVisualization

#Just generate a simple initial mesh for the disk
f(x, y) = x^2 + y^2 - 1 > 0 ? 1 : 0

TC = TriangulationCache(f;
    xlims = [-2, 2],
    ylims = [-2, 2],
    initial_resolution = 100,
    verbose = false,
)

fig = visualize(TC; refine_button = false)
display(fig)

TC = TriangulationCache(f;
           xlims = [-2, 2],
           ylims = [-2, 2],
           initial_resolution = 100,
           verbose = false,
           is_complete=(V,vals)->true,
           )

f(x,y) = x^2+y^2< 1 ? "inside" : "outside"           
TC = TriangulationCache(f;
           xlims = [-2, 2],
           ylims = [-2, 2],
           initial_resolution = 100,
           verbose = false,
           )
visualize(TC)