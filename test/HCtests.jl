using AdaptiveVisualization
using HomotopyContinuation
using LinearAlgebra
using Test

# This file is an opt-in HomotopyContinuation.jl example/test.
# It is intentionally not included from runtests.jl because it can take a while.
#
# Try it from the package environment with:
#
#     include("test/HCtests.jl")
#     F = TwentySevenLines()
#     f = real_solution_function(F)
#     TC, fig = visualize(f; xlims=[-5, 5], ylims=[-5, 5])
#
# Or run it directly by setting:
#
#     ADAPTIVE_SAMPLING_RUN_HC_EXAMPLES=true julia --project=. test/HCtests.jl

function TwentySevenLines()
    @var x y z
    @var a[1:4, 1:4, 1:4]

    terms = []
    for i in 0:3
        for j in 0:3
            for k in 0:3
                i + j + k <= 3 && push!(terms, [i, j, k])
            end
        end
    end

    f = sum(a[c[1]+1, c[2]+1, c[3]+1] * x^c[1] * y^c[2] * z^c[3] for c in terms)
    Params = [a[c[1]+1, c[2]+1, c[3]+1] for c in terms]

    @var t b[1:2] c[1:2]

    lx = t
    ly = b[1] * t + b[2]
    lz = c[1] * t + c[2]

    g = HomotopyContinuation.subs(f, [x, y, z] => [lx, ly, lz])
    Eqs = HomotopyContinuation.coefficients(g, [t])

    return System(Eqs; variables=[b[1], b[2], c[1], c[2]], parameters=Params)
end

# Plane conics in P^3 meeting points and lines; see Hauenstein-Sottile,
# "Algorithm XXX: alphaCertified", Section 5.3.
const SPACE_CONIC_DEGREES = Dict(
    (npoints=0, nlines=8) => 92,
    (npoints=1, nlines=6) => 18,
    (npoints=2, nlines=4) => 4,
    (npoints=3, nlines=2) => 1,
    (npoints=4, nlines=0) => 0,
)

function SpaceConics(; npoints::Int, nlines::Int)
    @assert 2npoints + nlines == 8 "Plane conics in P^3 have dimension 8, so need 2npoints + nlines == 8."
    @assert 0 <= npoints <= 4 "This Schubert problem has npoints in 0:4."
    @assert 0 <= nlines <= 8 "This Schubert problem has nlines in 0:8."

    @var A B C
    @var c11 c22 c12 c13 c23

    fixed_points = npoints >= 4 ? 4 : 0
    fixed_lines = fixed_points == 0 && nlines >= 3 ? 3 : 0
    free_points = npoints - fixed_points
    free_lines = nlines - fixed_lines

    # Kill the ambient projective symmetry in the parameter space by fixing
    # standard general-position incidence conditions whenever this Schubert
    # problem contains enough of them.
    e = [[i == j ? 1 : 0 for i in 1:4] for j in 1:4]

    # Free point parameters: p[:, i] is the i-th non-normalized point in P^3.
    @var p[1:4, 1:free_points]

    # Free line parameters: the j-th non-normalized line is spanned by u[:, j] and v[:, j].
    @var u[1:4, 1:free_lines] v[1:4, 1:free_lines]

    points = vcat(e[1:fixed_points], [p[:, i] for i in 1:free_points])
    # Use skew lines that are compatible with the conic chart q = ... + x3^2.
    # In particular, avoid fixing span(e3,e4), which would force the incidence
    # point [0:0:1] where this chart cannot see a conic.
    fixed_line_us = fixed_lines == 0 ? [] : [e[1], e[2], e[1] + e[2] + e[3]]
    fixed_line_vs = fixed_lines == 0 ? [] : [e[3], e[4], e[1] - e[2] + e[4]]
    line_us = vcat(fixed_line_us, [u[:, j] for j in 1:free_lines])
    line_vs = vcat(fixed_line_vs, [v[:, j] for j in 1:free_lines])

    variables = [A, B, C, c11, c22, c12, c13, c23]
    parameters = vcat(vec(p), vec(u), vec(v))

    q(x1, x2, x3) =
        c11*x1^2 +
        c22*x2^2 +
        x3^2 +
        c12*x1*x2 +
        c13*x1*x3 +
        c23*x2*x3

    h(x1, x2, x3, x4) = x4 - A*x1 - B*x2 - C*x3

    eqs = []

    for i in 1:npoints
        point = points[i]
        push!(eqs, h(point[1], point[2], point[3], point[4]))
        push!(eqs, q(point[1], point[2], point[3]))
    end

    for j in 1:nlines
        line_u = line_us[j]
        line_v = line_vs[j]

        hu = h(line_u[1], line_u[2], line_u[3], line_u[4])
        hv = h(line_v[1], line_v[2], line_v[3], line_v[4])

        z1 = hv*line_u[1] - hu*line_v[1]
        z2 = hv*line_u[2] - hu*line_v[2]
        z3 = hv*line_u[3] - hu*line_v[3]

        push!(eqs, q(z1, z2, z3))
    end

    return System(eqs; variables=variables, parameters=parameters)
end

# Compatibility with the original positional convention: lines first, points second.
SpaceConics(nlines::Int, npoints::Int) = SpaceConics(; npoints=npoints, nlines=nlines)

expected_space_conic_degree(; npoints::Int, nlines::Int) =
    SPACE_CONIC_DEGREES[(npoints=npoints, nlines=nlines)]

expected_space_conic_degree(nlines::Int, npoints::Int) =
    expected_space_conic_degree(; npoints=npoints, nlines=nlines)

function gram_schmidt(basis_vectors::Vector; kwargs...)
    M = hcat(basis_vectors...)
    Q = Matrix(qr(M).Q)
    return collect(eachcol(Q))
end

function restrict(F::System, P::Vector{Vector{Float64}})
    n = length(P)
    @var tt[1:n-1]

    basis_vectors = gram_schmidt([P[i] - P[1] for i in 2:n])
    affine_span = P[1] + sum(tt[i] .* basis_vectors[i] for i in 1:n-1)
    new_expressions = [subs(f, parameters(F) => affine_span) for f in expressions(F)]

    return System(new_expressions; variables=variables(F), parameters=tt)
end

function result_from_many_solve_item(item)
    return first(item)
end

function real_solution_function(
        F::System;
        plane_points = [randn(Float64, nparameters(F)) for _ in 1:3],
        start_parameters = nothing,
        backup_start_parameters = nothing,
        wildcard_parity_mismatch = true)

    G = restrict(F, plane_points)
    P = start_parameters === nothing ? randn(ComplexF64, nparameters(G)) : start_parameters
    P_backup = backup_start_parameters === nothing ? randn(ComplexF64, nparameters(G)) : backup_start_parameters
    S = solve(G; target_parameters=P)
    S_backup = solve(G; target_parameters=P_backup)
    total_solution_parity = isodd(nsolutions(S))

    function real_solution_counter(points)
        results = solve(G, S;
            start_parameters=P,
            target_parameters=points,
        )
        backup_results = nothing

        counts = Any[nreal(result_from_many_solve_item(R)) for R in results]
        if wildcard_parity_mismatch
            for i in eachindex(counts)
                isodd(counts[i]) == total_solution_parity && continue

                if backup_results === nothing
                    println("Parity mismatch detected; recalculating real solution counts with backup start solutions.")
                    backup_results = solve(G, S_backup;
                        start_parameters=P_backup,
                        target_parameters=points,
                    )
                end

                backup_count = nreal(result_from_many_solve_item(backup_results[i]))
                counts[i] = isodd(backup_count) == total_solution_parity ? backup_count : :wildcard
            end
        end
        return counts
    end

    return real_solution_counter
end

function run_twenty_seven_lines_example(;
        total_resolution=1000,
        initial_resolution_fraction=0.10,
        display_figure=true,
        kwargs...)
    @testset "Twenty-seven lines opt-in HC example" begin
        F = TwentySevenLines()
        f = real_solution_function(F)

        TC, fig = visualize(
            f;
            xlims=[-5, 5],
            ylims=[-5, 5],
            total_resolution=total_resolution,
            initial_resolution_fraction=initial_resolution_fraction,
            strategy=:sierpinski,
            kwargs...,
        )

        display_figure && display(fig)
        @test length(AdaptiveVisualization.function_values(TC)) > 0
        return TC
    end
end

if get(ENV, "ADAPTIVE_SAMPLING_RUN_HC_EXAMPLES", "false") == "true"
    run_twenty_seven_lines_example()
end
