include("test/HCtests.jl")
K = KuramotoModel(3)
f = real_solution_function(K)
TC = TriangulationCache(f;
                     xlims=[-.75,.75],
                     ylims=[-.75, .75],
                     strategy=:sierpinski,
                     verbose=false,
                     resolution=15^2,
                     )
using LaTeXStrings

V = visualize(TC; buttons=false, 
    edges = true, 
    show_legend=false,
       xlabel = "",
    ylabel = "",  
    xlabelsize = 40,
    ylabelsize = 40,    
    xticklabelsize = 0,
    yticklabelsize = 0,
    legend_title = "nreal",
    labels=false)
AdaptiveVisualization.save(V,"PaperFigures/K1.png")
refine!(TC)

V = visualize(TC; buttons=false, 
    edges = true, 
    show_legend=false,
       xlabel = "",
    ylabel = "",  
    xlabelsize = 40,
    ylabelsize = 40,    
    xticklabelsize = 0,
    yticklabelsize = 0,
    legend_title = "nreal",
    labels=false)
AdaptiveVisualization.save(V,"PaperFigures/K2.png")
refine!(TC)

V = visualize(TC; buttons=false, 
    edges = true, 
    show_legend=false,
       xlabel = "",
    ylabel = "",  
    xlabelsize = 40,
    ylabelsize = 40,    
    xticklabelsize = 0,
    yticklabelsize = 0,
    legend_title = "nreal",
    labels=false)
AdaptiveVisualization.save(V,"PaperFigures/K3.png")
refine!(TC)

V = visualize(TC; buttons=false, 
    edges = true, 
    show_legend=false,
       xlabel = "",
    ylabel = "",  
    xlabelsize = 40,
    ylabelsize = 40,    
    xticklabelsize = 0,
    yticklabelsize = 0,
    legend_title = "nreal",
    labels=false)
AdaptiveVisualization.save(V,"PaperFigures/K4.png")
refine!(TC)

V = visualize(TC; buttons=false, 
    edges = true, 
    show_legend=false,
       xlabel = "",
    ylabel = "",  
    xlabelsize = 40,
    ylabelsize = 40,    
    xticklabelsize = 0,
    yticklabelsize = 0,
    legend_title = "nreal",
    labels=false)
AdaptiveVisualization.save(V,"PaperFigures/K5.png")
refine!(TC)

V = visualize(TC; buttons=false, 
    edges = true, 
    show_legend=false,
       xlabel = "",
    ylabel = "",  
    xlabelsize = 40,
    ylabelsize = 40,    
    xticklabelsize = 0,
    yticklabelsize = 0,
    legend_title = "nreal",
    labels=false)
AdaptiveVisualization.save(V,"PaperFigures/K6.png")
