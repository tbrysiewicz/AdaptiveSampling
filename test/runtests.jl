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

    @testset "Invalid oracle arity" begin
        f3(x, y, z) = x + y + z
        @test_throws Exception ValuedSubdivision(f3; resolution=10)
    end
end
