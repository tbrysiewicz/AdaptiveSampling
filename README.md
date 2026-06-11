# AdaptiveVisualization.jl

AdaptiveVisualization.jl adaptively samples expensive functions over a two-dimensional
parameter window and visualizes the result with GLMakie. It is designed for
parameter landscapes where most regions are boring, but boundary regions or
jump loci deserve more samples.

![A triangulation cache with controls](docs/triangulation-cache.svg)

The Delaunay triangulation stores sampled points and triangle connectivity,
`TriangulationCache` stores oracle values by vertex index, and
`incomplete_triangles` stores only the triangles still needing refinement.
Refinement inserts new points only into incomplete triangles, using incremental
Delaunay updates.

## Installation

From a Julia REPL in this repository:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then load the package:

```julia
using AdaptiveVisualization
```

## Quick Start

```julia
using AdaptiveVisualization

f(x, y) = x^2 + y^2 < 1 ? 1 : 2

TC, fig = visualize(f;
    xlims = [-2, 2],
    ylims = [-2, 2],
    total_resolution = 1000,
    strategy = :sierpinski,
    refine_button = true,
)
```

`visualize(f; ...)` returns both the `TriangulationCache` and the Makie figure.
It also displays the figure. The `Refine` button performs another refinement
pass, and the arrow/zoom buttons move around the parameter window while keeping
previously computed function values available.

## Batched Oracles

The preferred oracle shape is batched:

```julia
f(points) = [p[1]^2 + p[2]^2 for p in points]
```

Single-point functions are also accepted:

```julia
f(point) = point[1]^2 + point[2]^2
g(x, y) = x^2 + y^2
```

Internally, single-point oracles are wrapped into batched calls. For expensive
applications, especially HomotopyContinuation.jl workflows, a real batched
oracle is usually much faster.

## Refinement

![The adaptive refinement loop](docs/refinement-loop.svg)

The main refinement strategies are:

- `:sierpinski`: add triangle edge midpoints that are not already cached.
- `:barycenter`: add the triangle barycenter.
- `:random`: add one random point inside the triangle.

By default, `refine!(TC)` performs one pass over visible incomplete triangles.
It skips triangles whose area is at or below
`TC.min_refinement_area * window_area`. The default normalized
`min_refinement_area` is `1e-4`.

To refine repeatedly until all remaining incomplete triangles are too small:

```julia
refine!(TC; by_min_area = 1e-5)
```

## Complete and Incomplete Triangles

A triangle is complete when its vertex values are consistent according to the
completeness predicate. Custom predicates receive the triangle vertices and the
three corresponding function values:

```julia
same_parity(vertices, values; kwargs...) = begin
    all(iseven, values) || all(isodd, values)
end

TC = TriangulationCache(f; is_complete = same_parity)
```

Here `vertices` is an `NTuple{3,NTuple{2,Float64}}`, and `values` is a vector
of the three oracle values at those vertices.

The default predicate treats non-real values as discrete, regardless of how many
distinct values are present. Discrete triangles are complete when all
non-`:wildcard` values are equal. Numeric triangles use a tolerance derived from
the global range of sampled real values. The special value `:wildcard` is
treated as compatible with every other value for completeness.

For plotting, numeric complete triangles are colored by the average of their
non-wildcard vertex values. Non-real complete triangles are colored by their
shared category; if their non-wildcard values are not equal, plotting raises an
error because there is no unambiguous category to draw. Triangles whose plotted
value is unknown or all-wildcard are drawn black.

## HomotopyContinuation.jl Use

A common use case is to count real solutions over a parameter plane. The helper
code in `test/HCtests.jl` shows one such workflow:

```julia
using LinearAlgebra
using HomotopyContinuation
using AdaptiveVisualization

include("test/HCtests.jl")

F = TwentySevenLines()
real_line_count = real_solution_function(F)

TC, fig = visualize(real_line_count;
    xlims = [-5, 5],
    ylims = [-5, 5],
    total_resolution = 1000,
    initial_resolution_fraction = 0.10,
    strategy = :sierpinski,
    refine_button = true,
)
```

The oracle produced by `real_solution_function` is batched and passes a list of
target parameters directly to `solve`.

## API

The main user-facing functions are:

- `visualize(f; kwargs...)`
- `TriangulationCache(f; kwargs...)`
- `refine!(TC; kwargs...)`
- `complete_triangles(TC)`
- `incomplete_triangles(TC)`
- `is_discrete(function_values)`
- `is_complete(vertices, values; kwargs...)` for custom completeness predicates
- `save(fig, filename; kwargs...)`

## License

AdaptiveVisualization.jl is released under the MIT license.
