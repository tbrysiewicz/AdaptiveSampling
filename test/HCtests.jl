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
