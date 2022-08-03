using Finch, SparseArrays, BenchmarkTools, Images, FileIO, FixedPointNumbers, Colors

include("TensorMarket.jl")
using .TensorMarket

function pngwrite(filename, I, V, shape)
    if length(shape) != 2
        error("Grayscale only")
    end

    out = Array{Gray{N0f8}, 2}(undef, shape[1],shape[2])

    for (coord, val) in zip(zip(I...), V)
        out[coord[1], coord[2]] = reinterpret(N0f8, val)
    end

    save(filename, out)
end

function alpha_opencv(B, C, alpha)
    as = Scalar{0.0, Float32}(alpha)
    mas = Scalar{0.0, Float32}(1- alpha)
    Bf = copyto!(f"ss"(zero(UInt8)), copy(rawview(channelview(B))))
    Cf = copyto!(f"ss"(zero(UInt8)), copy(rawview(channelview(C))))
    A_ref = copyto!(f"ss"(zero(UInt8)), copy(rawview(channelview(B))))

    f = x -> round(UInt8, x)

    @index @loop i j A_ref[i, j] = f(as[] * Bf[i, j] + mas[] * Cf[i, j])
    pngwrite("A_ref.png", ffindnz(A_ref)..., size(A_ref))
    
    @index @loop i j A_ref[i, j] = 0

    pngwrite("A.png", ffindnz(A_ref)..., size(A_ref))
    pngwrite("B.png", ffindnz(Bf)..., size(Bf))
    pngwrite("C.png", ffindnz(Cf)..., size(Cf))

    io = IOBuffer()

    run(pipeline(`./alpha_opencv A.png B.png C.png $alpha`, stdout=io))

    A = load("A.png")
    A_ref = load("A_ref.png")

    @assert A == A_ref

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function alpha_finch(B, C, alpha)
    as = Scalar{0.0, Float64}(alpha)
    mas = Scalar{0.0, Float64}(1- alpha)

    B = copyto!(f"ss"(0.0), B)
    C = copyto!(f"ss"(0.0), C)
    A = fiber(B)
    return @belapsed (A = $A; B=$B; C=$C; as=$as; mas=$mas; @index @loop i j A[i, j] = as[] * B[i, j] + mas[] * C[i, j])
end

B = load("/Users/danieldonenfeld/Developer/Finch-Proj/download_cache/sketches/pngs/1.png")
C = load("/Users/danieldonenfeld/Developer/Finch-Proj/download_cache/sketches/pngs/10001.png")

println("opencv_time: ", alpha_opencv(B, C, 0.5))
println("finch_time: ", alpha_finch(B, C, 0.5))