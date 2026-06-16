using AdaptiveVisualization
using GLMakie
using HomotopyContinuation
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "..", "test", "HCtests.jl"))

const ASSET_DIR = joinpath(@__DIR__, "assets")
const README_FIGURE_RESOLUTION = 144
const README_FIGURE_REFINEMENT_PASSES = 5
mkpath(ASSET_DIR)

function save_readme_figure(fig, filename)
    path = joinpath(ASSET_DIR, filename)
    GLMakie.save(path, fig; px_per_unit=2)
    println("Saved ", path)
    return path
end

function refine_readme_figure!(TC)
    for _ in 1:README_FIGURE_REFINEMENT_PASSES
        refine!(TC; verbose=false)
    end
    return TC
end

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

function disk_indicator_figure()
    f(x, y) = x^2 + y^2 < 1 ? 1 : 2
    TC = TriangulationCache(f;
        xlims=[-2, 2],
        ylims=[-2, 2],
        resolution=README_FIGURE_RESOLUTION,
        strategy=:sierpinski,
        verbose=false,
    )
    refine_readme_figure!(TC)
    return visualize(TC;
        refine_button=false,
        plot_triangle_edges=true,
        figure_size=(900, 700),
        legend_title="region",
    )
end

function categorical_figure()
    f(x, y) = x^2 + y^2 < 1 ? "inside" : "outside"
    TC = TriangulationCache(f;
        xlims=[-2, 2],
        ylims=[-2, 2],
        resolution=README_FIGURE_RESOLUTION,
        strategy=:sierpinski,
        verbose=false,
    )
    refine_readme_figure!(TC)
    return visualize(TC;
        refine_button=false,
        plot_triangle_edges=true,
        figure_size=(900, 700),
        legend_title="class",
    )
end

function kuramoto_figure()
    Random.seed!(3)
    F = KuramotoModel(3)
    f = real_solution_function(F)
    TC = TriangulationCache(f;
        xlims=[-1.5, 1.5],
        ylims=[-1.5, 1.5],
        resolution=README_FIGURE_RESOLUTION,
        strategy=:sierpinski,
        verbose=false,
    )
    refine_readme_figure!(TC)
    return visualize(TC;
        refine_button=false,
        plot_triangle_edges=true,
        figure_size=(900, 700),
        legend_title="nreal",
    )
end

const FIGURE_BUILDERS = Dict(
    "disk" => () -> save_readme_figure(disk_indicator_figure(), "disk-indicator.png"),
    "categorical" => () -> save_readme_figure(categorical_figure(), "categorical-inside-outside.png"),
    "kuramoto" => () -> save_readme_figure(kuramoto_figure(), "kuramoto-real-solutions.png"),
)

requested_figures = isempty(ARGS) ? ["disk", "categorical", "kuramoto"] : ARGS
for name in requested_figures
    haskey(FIGURE_BUILDERS, name) || error("Unknown README figure '$name'. Use one of $(sort(collect(keys(FIGURE_BUILDERS)))).")
    FIGURE_BUILDERS[name]()
end
