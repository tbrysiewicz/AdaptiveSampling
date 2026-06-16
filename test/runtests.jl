using Test
using AdaptiveVisualization

@testset "AdaptiveVisualization" begin
    @testset "Discrete/continuous detection" begin
        discrete_values = Any[1, 1, 2, :wildcard]
        continuous_values = [float(i) for i in 1:60]
        many_string_values = ["value-$i" for i in 1:60]

        @test is_discrete(discrete_values)
        @test !is_discrete(continuous_values)
        @test is_discrete(many_string_values)
        @test AdaptiveVisualization.values_are_complete(Any["a", :wildcard, "a"])
        @test !AdaptiveVisualization.values_are_complete(Any["a", :wildcard, "b"])
    end

    @testset "Batched TriangulationCache construction" begin
        batch_calls = Ref(0)
        batch_sizes = Int[]
        f(points) = begin
            batch_calls[] += 1
            push!(batch_sizes, length(points))
            [p[1] + p[2] for p in points]
        end

        TC = TriangulationCache(f;
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=16,
            verbose=false,
        )

        @test AdaptiveVisualization.dimension(TC) == 2
        @test length(AdaptiveVisualization.function_values(TC)) == 16
        @test batch_calls[] == 1
        @test first(batch_sizes) == 16
        @test length(complete_triangles(TC)) + length(incomplete_triangles(TC)) > 0
        @test AdaptiveVisualization.remaining_oracle_budget(TC) === nothing
    end

    @testset "Single-point oracle wrapping" begin
        TC = TriangulationCache((x, y) -> x^2 + y^2;
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=9,
            verbose=false,
        )

        @test length(AdaptiveVisualization.function_values(TC)) == 9
        @test all(value isa Real for value in AdaptiveVisualization.output_values(TC))
    end

    @testset "Wildcard completeness" begin
        TC = TriangulationCache(points -> fill(:wildcard, length(points));
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=9,
            verbose=false,
        )

        @test isempty(incomplete_triangles(TC))
        @test !isempty(complete_triangles(TC))
    end

    @testset "Refinement" begin
        never_complete(vertices, values; kwargs...) = false
        TC = TriangulationCache(points -> [p[1] - p[2] for p in points];
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=9,
            strategy=:barycenter,
            min_refinement_area=0.0,
            is_complete=never_complete,
            verbose=false,
        )

        before = length(AdaptiveVisualization.function_values(TC))
        inserted = refine!(TC; verbose=false)

        @test inserted > 0
        @test length(AdaptiveVisualization.function_values(TC)) > before
        @test TC.total_oracle_calls == length(AdaptiveVisualization.function_values(TC))
    end

    @testset "Refinement call budget" begin
        never_complete(vertices, values; kwargs...) = false
        TC = TriangulationCache(points -> [p[1] - p[2] for p in points];
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=9,
            strategy=:barycenter,
            min_refinement_area=0.0,
            is_complete=never_complete,
            verbose=false,
        )

        inserted = refine!(TC; budget=1, verbose=false)

        @test inserted == 1
        @test AdaptiveVisualization.remaining_oracle_budget(TC) === nothing

        TC.oracle_budget = 1
        inserted = refine!(TC; budget=2, verbose=false)

        @test inserted == 2
        @test AdaptiveVisualization.remaining_oracle_budget(TC) == 1
    end

    @testset "Iterative refinement by normalized area" begin
        never_complete(vertices, values; kwargs...) = false
        TC = TriangulationCache(points -> [0 for _ in points];
            xlims=[0, 2],
            ylims=[0, 3],
            resolution=9,
            strategy=:barycenter,
            min_refinement_area=1.0,
            is_complete=never_complete,
            verbose=false,
        )

        inserted = refine!(TC; by_min_area=0.25, verbose=false)

        @test inserted == 0
        @test TC.min_refinement_area == 0.25
        @test AdaptiveVisualization.scaled_min_refinement_area(TC) == 1.5
    end

    @testset "Max-area slider exponent mapping" begin
        TC = TriangulationCache(points -> [p[1] + p[2] for p in points];
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=9,
            max_refinement_area=4e-4,
            verbose=false,
        )

        slider_range = AdaptiveVisualization.default_max_area_slider_range()

        @test collect(slider_range) == collect(2:6)
        @test AdaptiveVisualization.default_max_area_slider_start(TC, collect(slider_range)) == 4
        @test isapprox(AdaptiveVisualization.max_area_from_slider_exponent(TC, 5), 4e-5)
        @test AdaptiveVisualization.max_area_slider_label(6) == "Max area: window * 1e-6"
    end

    @testset "Stable categorical color value order" begin
        TC = TriangulationCache(points -> [p[1] < 0 ? 5 : 9 for p in points];
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=9,
            verbose=false,
        )

        order = AdaptiveVisualization.stable_plot_value_order!(TC, [5, 9])
        @test order == [5, 9]

        order = AdaptiveVisualization.stable_plot_value_order!(TC, [3, 5])
        @test order == [5, 9, 3]
    end

    @testset "Categorical complete triangle plotting" begin
        always_complete(vertices, values; kwargs...) = true
        TC = TriangulationCache(points -> fill("inside", length(points));
            xlims=[-1, 1],
            ylims=[-1, 1],
            resolution=4,
            is_complete=always_complete,
            verbose=false,
        )
        triangle = first(complete_triangles(TC))

        @test AdaptiveVisualization.triangle_plot_value(TC, triangle) == "inside"

        AdaptiveVisualization.function_values(TC)[triangle[1]] = "outside"
        @test_throws ErrorException AdaptiveVisualization.triangle_plot_value(TC, triangle)
    end

    @testset "Example visualization workflows" begin
        disk_indicator(x, y) = x^2 + y^2 < 1 ? 1 : 0
        TC = TriangulationCache(disk_indicator;
            xlims=[-2, 2],
            ylims=[-2, 2],
            resolution=25,
            verbose=false,
        )

        fig = visualize(TC; refine_button=false)
        @test fig isa AdaptiveVisualization.GLMakie.Figure

        fig = visualize(TC; refine_button=true, plot_triangle_edges=true)
        @test fig isa AdaptiveVisualization.GLMakie.Figure

        complete_TC = TriangulationCache(disk_indicator;
            xlims=[-2, 2],
            ylims=[-2, 2],
            resolution=25,
            verbose=false,
            is_complete=(vertices, values) -> true,
        )
        @test isempty(incomplete_triangles(complete_TC))
        @test !isempty(complete_triangles(complete_TC))

        region_label(x, y) = x^2 + y^2 < 1 ? "inside" : "outside"
        categorical_TC = TriangulationCache(region_label;
            xlims=[-3, 3],
            ylims=[-2, 2],
            resolution=25,
            verbose=false,
        )
        @test is_discrete(AdaptiveVisualization.output_values(categorical_TC))

        fig = visualize(categorical_TC; refine_button=false, plot_triangle_edges=true)
        @test fig isa AdaptiveVisualization.GLMakie.Figure
    end

    @testset "Invalid strategy" begin
        @test_throws Exception TriangulationCache((x, y) -> x + y; strategy=:quadtree, verbose=false)
        @test_throws Exception TriangulationCache((x, y) -> x + y; initial_resolution=9, verbose=false)
        @test_throws Exception TriangulationCache((x, y) -> x + y; total_resolution=9, verbose=false)
    end
end
