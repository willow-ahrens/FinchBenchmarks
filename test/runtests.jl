using TensorDepot
using MatrixDepot
using Test

ENV["DATADEPS_ALWAYS_ACCEPT"] = true

@testset "TensorDepot.jl" begin
    @test size(matrixdepot("humansketches", 1:10)) == (10,1111,1111)

    if "local" in ARGS
        @test size(matrixdepot("humansketches")) == (20000,1111,1111)
    end

    @test_throws BoundsError matrixdepot("humansketches", 100000:1000001)

    if "local" in ARGS
        @test size(matrixdepot("spgemm")) == (241600,18)
    end
end