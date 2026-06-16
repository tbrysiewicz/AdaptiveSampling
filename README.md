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
)
```

`visualize(f; ...)` returns both the `TriangulationCache` and the Makie figure.
It also displays the figure. By default, one quarter of `total_resolution` is
used for initialization and the rest is used for adaptive refinement; pass
`initial_resolution` to choose the initialization mesh size directly. The
`Refine` button performs another refinement pass, and the arrow/zoom buttons
move around the parameter window while keeping previously computed function
values available.

<img src="docs/assets/disk-indicator.png" alt="Adaptive sampling of the disk indicator quick-start function" width="450">

## Pipeline

![Flowchart for initializing a TriangulationCache](docs/initialization-flow.svg)

Flowchart for initializing a TriangulationCache

![Flowchart for adaptive refinement](docs/refinement-loop.svg)

Each refinement pass inserts points in incomplete triangles, evaluates the
function at the new points, updates the Delaunay triangulation, and classifies
the resulting triangles as complete or incomplete.

## Kuramoto Example

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

Regenerate the README figures with:

```julia
julia --project=. docs/make_readme_figures.jl
```

AdaptiveVisualization.jl is released under the MIT license.
