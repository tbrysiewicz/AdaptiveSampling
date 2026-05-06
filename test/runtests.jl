using Test
using AdaptiveSampling

@testset "AdaptiveSampling" begin
    @testset "Discrete/continuous detection" begin
        discrete_cache = [([0.0, 0.0], value) for value in (1, 1, 2, :wildcard)]
        continuous_cache = [([float(i), 0.0], float(i)) for i in 1:60]

        @test is_discrete(discrete_cache)
        @test !is_discrete(continuous_cache)
    end

    @testset "Batched ValuedTriangulation construction" begin
        batch_calls = Ref(0)
        batch_sizes = Int[]
        f(points) = begin
            batch_calls[] += 1
            push!(batch_sizes, length(points))
            [p[1] + p[2] for p in points]
        end

        VT = ValuedTriangulation(f;
            xlims=[-1, 1],
            ylims=[-1, 1],
            total_resolution=64,
            initial_resolution=16,
            verbose=false,
        )

        @test AdaptiveSampling.dimension(VT) == 2
        @test length(AdaptiveSampling.function_cache(VT)) == 16
        @test batch_calls[] == 1
        @test first(batch_sizes) == 16
        @test length(complete_polygons(VT)) + length(incomplete_polygons(VT)) > 0
        @test AdaptiveSampling.remaining_oracle_budget(VT) === nothing
    end

    @testset "Single-point oracle wrapping" begin
        VT = ValuedTriangulation((x, y) -> x^2 + y^2;
            xlims=[-1, 1],
            ylims=[-1, 1],
            initial_resolution=9,
            verbose=false,
        )

        @test length(AdaptiveSampling.function_cache(VT)) == 9
        @test all(value isa Real for value in AdaptiveSampling.output_values(VT))
    end

    @testset "Wildcard completeness" begin
        VT = ValuedTriangulation(points -> fill(:wildcard, length(points));
            xlims=[-1, 1],
            ylims=[-1, 1],
            initial_resolution=9,
            verbose=false,
        )

        @test isempty(incomplete_polygons(VT))
        @test !isempty(complete_polygons(VT))
    end

    @testset "Refinement" begin
        never_complete(triangle, function_cache; kwargs...) = false
        VT = ValuedTriangulation(points -> [p[1] - p[2] for p in points];
            xlims=[-1, 1],
            ylims=[-1, 1],
            initial_resolution=9,
            strategy=:barycenter,
            min_refinement_area=0.0,
            is_complete=never_complete,
            verbose=false,
        )

        before = length(AdaptiveSampling.function_cache(VT))
        inserted = refine!(VT; verbose=false)

        @test inserted > 0
        @test length(AdaptiveSampling.function_cache(VT)) > before
        @test VT.total_oracle_calls == length(AdaptiveSampling.function_cache(VT))
    end

    @testset "Iterative refinement by normalized area" begin
        never_complete(triangle, function_cache; kwargs...) = false
        VT = ValuedTriangulation(points -> [0 for _ in points];
            xlims=[0, 2],
            ylims=[0, 3],
            initial_resolution=9,
            strategy=:barycenter,
            min_refinement_area=1.0,
            is_complete=never_complete,
            verbose=false,
        )

        inserted = refine!(VT; by_min_area=0.25, verbose=false)

        @test inserted == 0
        @test VT.min_refinement_area == 0.25
        @test AdaptiveSampling.scaled_min_refinement_area(VT) == 1.5
    end

    @testset "Max-area slider exponent mapping" begin
        VT = ValuedTriangulation(points -> [p[1] + p[2] for p in points];
            xlims=[-1, 1],
            ylims=[-1, 1],
            initial_resolution=9,
            max_refinement_area=4e-4,
            verbose=false,
        )

        slider_range = AdaptiveSampling.default_max_area_slider_range()

        @test collect(slider_range) == collect(2:6)
        @test AdaptiveSampling.default_max_area_slider_start(VT, collect(slider_range)) == 4
        @test isapprox(AdaptiveSampling.max_area_from_slider_exponent(VT, 5), 4e-5)
        @test AdaptiveSampling.max_area_slider_label(6) == "Max area: window * 1e-6"
    end

    @testset "Stable categorical color value order" begin
        VT = ValuedTriangulation(points -> [p[1] < 0 ? 5 : 9 for p in points];
            xlims=[-1, 1],
            ylims=[-1, 1],
            initial_resolution=9,
            verbose=false,
        )

        order = AdaptiveSampling.stable_plot_value_order!(VT, [5, 9])
        @test order == [5, 9]

        order = AdaptiveSampling.stable_plot_value_order!(VT, [3, 5])
        @test order == [5, 9, 3]
    end

    @testset "Invalid strategy" begin
        @test_throws Exception ValuedTriangulation((x, y) -> x + y; strategy=:quadtree, verbose=false)
    end
end
