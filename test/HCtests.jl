using AdaptiveSampling
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
#     VT, fig = visualize(f; xlims=[-5, 5], ylims=[-5, 5])
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

    # Point parameters: p[:, i] is the i-th point in P^3.
    @var p[1:4, 1:npoints]

    # Line parameters: the j-th line is spanned by u[:, j] and v[:, j].
    @var u[1:4, 1:nlines] v[1:4, 1:nlines]

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
        push!(eqs, h(p[1, i], p[2, i], p[3, i], p[4, i]))
        push!(eqs, q(p[1, i], p[2, i], p[3, i]))
    end

    for j in 1:nlines
        hu = h(u[1, j], u[2, j], u[3, j], u[4, j])
        hv = h(v[1, j], v[2, j], v[3, j], v[4, j])

        z1 = hv*u[1, j] - hu*v[1, j]
        z2 = hv*u[2, j] - hu*v[2, j]
        z3 = hv*u[3, j] - hu*v[3, j]

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
        wildcard_parity_mismatch = true)

    G = restrict(F, plane_points)
    P = start_parameters === nothing ? randn(ComplexF64, nparameters(G)) : start_parameters
    S = solve(G; target_parameters=P)
    total_solution_parity = isodd(nsolutions(S))

    function real_solution_counter(points)
        results = solve(G, S;
            start_parameters=P,
            target_parameters=points,
        )

        counts = [nreal(result_from_many_solve_item(R)) for R in results]
        if wildcard_parity_mismatch
            return [isodd(c) == total_solution_parity ? c : :wildcard for c in counts]
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

        VT, fig = visualize(
            f;
            xlims=[-5, 5],
            ylims=[-5, 5],
            total_resolution=total_resolution,
            initial_resolution_fraction=initial_resolution_fraction,
            strategy=:sierpinski,
            kwargs...,
        )

        display_figure && display(fig)
        @test length(AdaptiveSampling.function_cache(VT)) > 0
        return VT
    end
end

if get(ENV, "ADAPTIVE_SAMPLING_RUN_HC_EXAMPLES", "false") == "true"
    run_twenty_seven_lines_example()
end
