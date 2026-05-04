using Test
using AdaptiveSampling

@testset "AdaptiveSampling basics" begin
    @testset "Discrete/continuous detection" begin
        fc_discrete = [([0.0, 0.0], 1), ([1.0, 0.0], 1), ([0.0, 1.0], 2)]
        fc_continuous = [([float(i), 0.0], float(i)) for i in 1:60]
        @test is_discrete(fc_discrete)
        @test !is_discrete(fc_continuous)
    end

    @testset "2D subdivision construction and refinement" begin
        f2(x, y) = x + y
        vsd = ValuedSubdivision(f2; xlims=[-1, 1], ylims=[-1, 1], resolution=100, strategy=:quadtree)

        @test AdaptiveSampling.dimension(vsd) == 2
        @test length(AdaptiveSampling.function_cache(vsd)) > 0
        @test length(complete_polygons(vsd)) + length(incomplete_polygons(vsd)) > 0

        n_before = length(AdaptiveSampling.function_cache(vsd))
        refine!(vsd, 10; strategy=:quadtree)
        @test length(AdaptiveSampling.function_cache(vsd)) >= n_before
        @test length(unique(AdaptiveSampling.input_points(vsd))) == length(AdaptiveSampling.input_points(vsd))
    end

    @testset "1D oracle auto-strategy" begin
        f1(x) = x^2
        vsd1 = ValuedSubdivision(f1; xlims=[-1, 1], resolution=50)

        @test AdaptiveSampling.dimension(vsd1) == 1
        @test all(length(p) == 2 for p in vcat(complete_polygons(vsd1), incomplete_polygons(vsd1)))

        n_before = length(AdaptiveSampling.function_cache(vsd1))
        refine!(vsd1, 5)
        @test length(AdaptiveSampling.function_cache(vsd1)) >= n_before
    end

    @testset "Batched oracle support" begin
        batch_calls = Ref(0)
        batch_sizes = Int[]
        f_batch(points) = begin
            batch_calls[] += 1
            push!(batch_sizes, length(points))
            [p[1] + p[2] for p in points]
        end

        vsd = ValuedSubdivision(f_batch; xlims=[-1, 1], ylims=[-1, 1], resolution=25, strategy=:quadtree)

        @test AdaptiveSampling.dimension(vsd) == 2
        @test batch_calls[] == 1
        @test first(batch_sizes) == length(AdaptiveSampling.function_cache(vsd))

        n_before = length(AdaptiveSampling.function_cache(vsd))
        refine!(vsd, 5; strategy=:quadtree)
        @test length(AdaptiveSampling.function_cache(vsd)) > n_before
        @test batch_calls[] >= 2
        @test batch_sizes[end] > 1
    end

    @testset "Explicit 1D batched oracle" begin
        f_batch_1d(points) = [p[1]^2 for p in points]
        vsd = ValuedSubdivision(f_batch_1d; xlims=[-1, 1], resolution=20, strategy=:onedimensional)

        @test AdaptiveSampling.dimension(vsd) == 1
        @test all(length(p) == 2 for p in vcat(complete_polygons(vsd), incomplete_polygons(vsd)))
    end

    @testset "Refinement without complete polygons" begin
        never_complete(p, FC; kwargs...) = false
        f2(x, y) = x - y
        vsd = ValuedSubdivision(f2; xlims=[-1, 1], ylims=[-1, 1], resolution=25, strategy=:barycentric, is_complete=never_complete)

        @test isempty(complete_polygons(vsd))
        n_before = length(AdaptiveSampling.function_cache(vsd))
        refine!(vsd, 1)
        @test length(AdaptiveSampling.function_cache(vsd)) == n_before + 1
    end

    @testset "Strategy inference follows current polygon shape" begin
        never_complete(p, FC; kwargs...) = false
        f2(x, y) = x - y

        quad_vsd = ValuedSubdivision(f2; xlims=[-1, 1], ylims=[-1, 1], resolution=25, strategy=:quadtree, is_complete=never_complete)
        delaunay_retriangulate!(quad_vsd)
        @test all(length(p) == 3 for p in incomplete_polygons(quad_vsd))

        n_before = length(AdaptiveSampling.function_cache(quad_vsd))
        refine!(quad_vsd, 1)
        @test length(AdaptiveSampling.function_cache(quad_vsd)) == n_before + 3
        @test_throws Exception refine!(quad_vsd, 1; strategy=:quadtree)

        sierpinski_default = ValuedSubdivision(f2; xlims=[-1, 1], ylims=[-1, 1], resolution=25, strategy=:sierpinski, is_complete=never_complete)
        n_before = length(AdaptiveSampling.function_cache(sierpinski_default))
        refine!(sierpinski_default, 1)
        @test length(AdaptiveSampling.function_cache(sierpinski_default)) == n_before + 3

        sierpinski_explicit = ValuedSubdivision(f2; xlims=[-1, 1], ylims=[-1, 1], resolution=25, strategy=:sierpinski, is_complete=never_complete)
        n_before = length(AdaptiveSampling.function_cache(sierpinski_explicit))
        refine!(sierpinski_explicit, 1; strategy=:sierpinski)
        @test length(AdaptiveSampling.function_cache(sierpinski_explicit)) == n_before + 3
    end

    @testset "Careful refinement retriangulates after adding points" begin
        never_complete(p, FC; kwargs...) = false
        f2(x, y) = x - y

        vsd = ValuedSubdivision(f2; xlims=[-1, 1], ylims=[-1, 1], resolution=25, strategy=:quadtree, is_complete=never_complete)
        refine!(vsd, 4; strategy=:careful)

        @test all(length(p) == 3 for p in incomplete_polygons(vsd))
        @test all(length(unique(p)) == 3 for p in incomplete_polygons(vsd))
        @test length(unique(AdaptiveSampling.input_points(vsd))) == length(AdaptiveSampling.input_points(vsd))
        @test AdaptiveSampling.strategy(vsd) == :careful
    end

    @testset "Visualization edge cases" begin
        zero_vsd = ValuedSubdivision((x, y) -> 0.0; xlims=[-1, 1], ylims=[-1, 1], resolution=16, strategy=:quadtree)
        @test visualize(zero_vsd; plot_all_polygons=true) !== nothing

        nonnumeric_vsd = ValuedSubdivision((x, y) -> "region"; xlims=[-1, 1], ylims=[-1, 1], resolution=16, strategy=:quadtree)
        @test visualize(nonnumeric_vsd; plot_all_polygons=true) !== nothing

        one_d_vsd = ValuedSubdivision(x -> 0.0; xlims=[-1, 1], resolution=8)
        @test visualize(one_d_vsd; plot_all_polygons=true) !== nothing
    end

    @testset "Invalid oracle arity" begin
        f3(x, y, z) = x + y + z
        @test_throws Exception ValuedSubdivision(f3; resolution=10)
    end
end
