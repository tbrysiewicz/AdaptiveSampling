# AdaptiveVisualization.jl

AdaptiveVisualization.jl adaptively samples expensive functions over a two-dimensional
parameter window and visualizes the result with GLMakie. It is designed for
parameter landscapes where most regions are boring, but boundary regions or
jump loci deserve more samples.

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
    buttons = true,
    legend_title = "value",
    title = "Continuous scalar field",
)
```

`visualize(f; ...)` returns both the `TriangulationCache` and the Makie figure.
It also displays the figure. By default, one quarter of `total_resolution` is
used for initialization and the rest is used for adaptive refinement; pass
`initial_resolution` to choose the initialization mesh size directly. The
`Refine` button performs another refinement pass, and the arrow/zoom buttons
move around the parameter window while keeping previously computed function
values available.

<p align="center">
  <img src="docs/assets/disk-indicator.png" alt="Adaptive sampling of the disk indicator quick-start function" width="450">
</p>

## HomotopyContinuation

Load HomotopyContinuation alongside AdaptiveVisualization to visualize a
polynomial system directly. The integration activates automatically; install
HomotopyContinuation in your active environment if needed with
`Pkg.add("HomotopyContinuation")`.

```julia
using AdaptiveVisualization
using HomotopyContinuation

@var x y a b
F = System([x^2 + y^2 - a, x - y + b^3]; variables=[x, y], parameters=[a, b])

TC, fig = visualize(F)
TC, fig = visualize(F; near=[2.3123, 0.0])
TC, fig = visualize(F; plane_points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
TC, fig = visualize(F; func=:real)          # Default: numerical real-solution count
TC, fig = visualize(F; func=:certify_real)  # Optional soft certificates
TC, fig = visualize(F; func=:positive)     # All variable coordinates positive
TC, fig = visualize(F; func=:dietmaier)    # Minimum nonzero imaginary L1 norm
```

`visualize(F; ...)` returns `(TC, fig)` and accepts the usual visualization and
refinement keywords. Systems must have at least two parameters. The same plane
keywords apply to every HC evaluator:

- `near=p` centers the view at `p`. For two parameters, it keeps the original
  coordinate axes. For more parameters, it completes the plane points as
  `[p, p + v1, p + v2]` with random orthonormal directions `v1`, `v2`.
- Three `plane_points=[p, q, r]` override `near` completely. Supply real, finite
  coordinates in the order returned by `parameters(F)`, with independent
  directions `q - p` and `r - p`; their lengths are used as supplied.
- With neither keyword, a two-parameter system uses its original parameter
  coordinates, centered at zero. Systems with more parameters use a random real
  center and plane. Pass a seeded `rng`, such as `MersenneTwister(42)` from
  `Random`, to reproduce random choices.

Displayed coordinates `(u, v)` map to parameters by
`p + zoomer * (u * (q - p) + v * (r - p))`. Thus `(0, 0)` represents `p`;
`xlims` and `ylims` set the displayed window, and `zoomer` scales it in parameter
space. For example, `near=p, zoomer=0.01` explores a small neighborhood of `p`.
For a two-parameter system, the default is equivalent to
`plane_points=[[0, 0], [1, 0], [0, 1]]`, in the order returned by `parameters(F)`.

| `func` | Value at each sampled parameter point |
| --- | --- |
| `:certify_real` | Real-solution count using HC certification and checks for nonreal solutions. |
| `:real` (default) | Numerical real-solution count, with no certification. |
| `:positive` | Numerical count of real solutions with every variable strictly positive. |
| `:dietmaier` | Minimum imaginary L1 norm exceeding `imaginary_zero_atol`, or zero if none does. |

`:certify_real` produces **soft certificates because its input is floating
point**. These describe the system supplied numerically to HC, rather than an
exact symbolic guarantee for the intended coefficients. The evaluator checks
distinct certified solutions, classifies real and nonreal solutions, and retries
unresolved samples. Samples that remain unresolved are represented by `:wildcard`.

The numerical counting modes use `real_tol` to classify solutions as real. `:positive`
also requires every real coordinate to exceed `positivity_tol` (default `0.0`);
for example, `positivity_tol=1e-8` excludes coordinates at or below that threshold.
The parity check applies to the total real count, never to the positive subset.
`max_retries` controls additional attempts: five by default for `:certify_real`,
two for numerical modes. Retry start solutions are prepared only when needed and
reused across batches. Set `verbose=true` for diagnostics.

Reusable evaluators prepare start solutions once and accept batches of displayed
coordinates:

```julia
plane = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
f = real_solution_function(F; plane_points=plane)
counts = f([[1.0, 0.0], [-1.0, 0.0]])  # [2, 0]
TC, fig = visualize(f; xlims=[-1, 3], ylims=[-1, 1])

f_certified = certify_real(F; near=[1.0, 0.0])
f_positive = positive_solution_function(F; plane_points=plane)
```

To reuse a known generic fiber, provide `start_parameters` and `start_solutions`
together. These starting parameters are independent of `near`. Advanced HC
options can be passed as named tuples through `solver_options`,
`monodromy_options`, and `certification_options`. The continuous
`dietmaier_function` evaluator is described below.

The `EXAMPLES` block at the end of
[ext/HomotopyContinuationExt.jl](ext/HomotopyContinuationExt.jl) contains the
`kuramoto_model`, `TwentySevenLines`, and `SpaceConics` constructors, the
`expected_space_conic_degree` helper, and `run_twenty_seven_lines_example`.
`kuramoto_model` is exported by AdaptiveVisualization; its implementation becomes
available when HomotopyContinuation is loaded. The remaining helpers belong to
the extension module. Loading the packages does not run examples or open figures.
Access the remaining helpers through the module:

```julia
HCExamples = Base.get_extension(AdaptiveVisualization, :HomotopyContinuationExt)
F = HCExamples.TwentySevenLines()
TC, fig = visualize(F; func=:real)
```

Cubic-surface and conic examples can take substantially longer than the small
polynomial example above. The legacy `test/HCtests.jl` path makes the example
helper names available for existing scripts.

## Pipeline

Flowchart for initializing a TriangulationCache: 

![Flowchart for initializing a TriangulationCache](docs/initialization-flow.svg)

Each refinement pass inserts points in incomplete triangles, evaluates the
function at the new points, updates the Delaunay triangulation, and classifies
the resulting triangles as complete or incomplete.

![Flowchart for adaptive refinement](docs/refinement-loop.svg)


## Kuramoto Example

```julia
using AdaptiveVisualization
using HomotopyContinuation

TC, fig = visualize(kuramoto_model(3);
    func=:real,
    plane_points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
    xlims=[-1, 1],
    ylims=[-1, 1],
)
```

The grid below shows the real-solution count for the `n = 3` Kuramoto model.
Columns are initialization resolutions `25`, `100`, `1600`, `2500`, and
`10000`. Rows are refinement steps `r = 0, ..., 5`. The image in row `r`,
column `n` is the result of visualizing the real solution function after
initialization resolution `n` and `r` refinement passes.

<table>
  <thead>
    <tr>
      <th><code>r \ n</code></th>
      <th><code>5^2</code></th>
      <th><code>10^2</code></th>
      <th><code>40^2</code></th>
      <th><code>50^2</code></th>
      <th><code>100^2</code></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th><code>0</code></th>
      <td><img src="docs/assets/kuramoto-n25-r0.png" alt="(25, 0)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n100-r0.png" alt="(100, 0)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n1600-r0.png" alt="(1600, 0)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n2500-r0.png" alt="(2500, 0)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n10000-r0.png" alt="(10000, 0)" width="150"></td>
    </tr>
    <tr>
      <th><code>1</code></th>
      <td><img src="docs/assets/kuramoto-n25-r1.png" alt="(25, 1)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n100-r1.png" alt="(100, 1)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n1600-r1.png" alt="(1600, 1)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n2500-r1.png" alt="(2500, 1)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n10000-r1.png" alt="(10000, 1)" width="150"></td>
    </tr>
    <tr>
      <th><code>2</code></th>
      <td><img src="docs/assets/kuramoto-n25-r2.png" alt="(25, 2)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n100-r2.png" alt="(100, 2)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n1600-r2.png" alt="(1600, 2)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n2500-r2.png" alt="(2500, 2)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n10000-r2.png" alt="(10000, 2)" width="150"></td>
    </tr>
    <tr>
      <th><code>3</code></th>
      <td><img src="docs/assets/kuramoto-n25-r3.png" alt="(25, 3)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n100-r3.png" alt="(100, 3)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n1600-r3.png" alt="(1600, 3)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n2500-r3.png" alt="(2500, 3)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n10000-r3.png" alt="(10000, 3)" width="150"></td>
    </tr>
    <tr>
      <th><code>4</code></th>
      <td><img src="docs/assets/kuramoto-n25-r4.png" alt="(25, 4)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n100-r4.png" alt="(100, 4)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n1600-r4.png" alt="(1600, 4)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n2500-r4.png" alt="(2500, 4)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n10000-r4.png" alt="(10000, 4)" width="150"></td>
    </tr>
    <tr>
      <th><code>5</code></th>
      <td><img src="docs/assets/kuramoto-n25-r5.png" alt="(25, 5)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n100-r5.png" alt="(100, 5)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n1600-r5.png" alt="(1600, 5)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n2500-r5.png" alt="(2500, 5)" width="150"></td>
      <td><img src="docs/assets/kuramoto-n10000-r5.png" alt="(10000, 5)" width="150"></td>
    </tr>
  </tbody>
</table>

## Non-discrete Functions

The sampled function does not need to be discrete or categorical. Continuous
real-valued functions are visualized with interpolated triangle colors, so the
same refinement workflow can be used for scalar fields such as:

```julia
f = (x, y) -> x^2 + y^2 - x

TC, fig = visualize(f;
    xlims = [-2, 2],
    ylims = [-2, 2],
    total_resolution = 1000,
    strategy = :sierpinski,
    buttons = true,
)
```

<p align="center">
  <img src="docs/assets/continuous-quadratic.png" alt="Adaptive sampling of x^2 + y^2 - x" width="450">
</p>

For polynomial systems, the oracle can also return a continuous statistic of
the complex solutions. The `dietmaier_function` oracle solves the system at
each sampled parameter point, computes the L1 norm of the imaginary parts of
each complex solution, discards values below a numerical zero tolerance, and
returns the minimum remaining norm. Small values indicate parameters where at
least one solution is nearly real. Set `imaginary_zero_atol` to choose the zero
tolerance; the evaluator returns zero if no nonzero imaginary norm remains.
Use it directly as `visualize(F; func=:dietmaier, imaginary_zero_atol=1e-10)`
or construct a reusable `dietmaier_function(F; ...)` evaluator.

For the `n = 3` Kuramoto model:

```julia
using Random
Random.seed!(3)
F = kuramoto_model(3)
f = dietmaier_function(F;
    plane_points=[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
)

TC, fig = visualize(f;
    xlims = [-1, 1],
    ylims = [-1, 1],
    total_resolution = 1000,
    strategy = :sierpinski,
    legend_title = "minimum nonzero imaginary L1 norm",
    title = "Kuramoto: Dietmaier Function",
    plot_log_transform = true,
)
```

<p align="center">
  <img src="docs/assets/dietmaier-kuramoto.png" alt="Adaptive sampling of the Kuramoto imaginary-part L1 norm" width="450">
</p>

Regenerate the README figures with:

```julia
julia --project=. docs/make_readme_figures.jl
```

AdaptiveVisualization.jl is released under the MIT license.
