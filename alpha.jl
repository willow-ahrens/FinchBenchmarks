using Finch, SparseArrays, BenchmarkTools, Images, FileIO, FixedPointNumbers, Colors
using JSON
using TensorDepot, MatrixDepot

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
    Bf = copyto!(@f(s(s(e($(zero(UInt8)))))), copy(rawview(channelview(B))))
    Cf = copyto!(@f(s(s(e($(zero(UInt8)))))), copy(rawview(channelview(C))))
    A_ref = copyto!(@f(s(s(e($(zero(UInt8)))))), copy(rawview(channelview(B))))

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

function ffindrepeats(src)
    rows = []
    cols = []
    for i in 1:src.lvl.I
        j = 1
        for jpos in src.lvl.lvl.pos[i]:(src.lvl.lvl.pos[i+1]-1)
            push!(rows, i)
            push!(cols, j)
            j = src.lvl.lvl.idx[jpos]
        end
    end
    return ((rows,cols), src.lvl.lvl.val[1:length(rows)])
end

function alpha_taco_rle(B, C, alpha)
    as = Scalar{0.0, Float32}(alpha)
    mas = Scalar{0.0, Float32}(1- alpha)
    Bf = copyto!(f"sr"(zero(UInt8)), copy(rawview(channelview(B))))
    Cf = copyto!(f"sr"(zero(UInt8)), copy(rawview(channelview(C))))
    A_ref = copyto!(f"sr"(zero(UInt8)), copy(rawview(channelview(B))))

    f = x -> round(UInt8, x)

    @index @loop i j A_ref[i, j] = f(as[] * Bf[i, j] + mas[] * Cf[i, j])
    ttwrite("A_ref.ttx", ffindrepeats(A_ref)..., size(A_ref))
    
    @index @loop i j A_ref[i, j] = 0

    ttwrite("A.ttx", ffindrepeats(A_ref)..., size(A_ref))
    ttwrite("B.ttx", ffindrepeats(Bf)..., size(Bf))
    ttwrite("C.ttx", ffindrepeats(Cf)..., size(Cf))

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./taco-rle/build/lib", "LD_LIBRARY_PATH" => "./taco-rle/build/lib") do
        run(pipeline(`./alpha_taco_rle A.ttx B.ttx C.ttx $alpha`, stdout=io))
    end
    
    A = ttread("A.ttx")
    A_ref = ttread("A_ref.ttx")

    @assert A == A_ref

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function alpha_finch(B, C, alpha)
    as = Scalar{0.0, Float64}(alpha)
    mas = Scalar{0.0, Float64}(1- alpha)

    B = copyto!(@f(s(r((zero(UInt8))))), copy(rawview(channelview(B))))
    C = copyto!(@f(s(r((zero(UInt8))))), copy(rawview(channelview(C))))
    A = fiber(B)
    f = x -> round(UInt8, x)
    return @belapsed (A = $A; B=$B; C=$C; as=$as; mas=$mas; f=$f; @index @loop i j A[i, j] = f(as[] * B[i, j] + mas[] * C[i, j]))
end

function alpha_finch_sparse(B, C, alpha)
    as = Scalar{0.0, Float64}(alpha)
    mas = Scalar{0.0, Float64}(1- alpha)

    B = copyto!(@f(s(r((one(UInt8))))), copy(rawview(channelview(B))))
    C = copyto!(@f(s(r((one(UInt8))))), copy(rawview(channelview(C))))

    A = fiber(B)
    f = x -> round(UInt8, x)
    return @belapsed (A = $A; B=$B; C=$C; as=$as; mas=$mas; f=$f; @index @loop i j A[i, j] = f(as[] * B[i, j] + mas[] * C[i, j]))
end

kernel_str = "@index @loop i j round(UInt8, A[i, j] = as[] * B[i, j] + mas[] * C[i, j])"
alpha = 0.5

numSketches = 2
humansketchesA = matrixdepot("humansketches", 1:numSketches)
humansketchesB = matrixdepot("humansketches", (numSketches+1):(10000+numSketches))

run(pipeline(`make alpha_opencv`))

results = Vector{Dict{String, <: Any}}()
for i in 1:numSketches 
    println("Performing op: $i")
    B = humansketchesA[i, :, :]
    C = humansketchesB[i, :, :]

    opencvResult = alpha_opencv(B, C, 0.5)
    push!(results, Dict("kernel"=>kernel_str, "alpha"=>alpha,"kind"=>"opencv","time"=>opencvResult,"dataset"=>"humansketches","imageB"=>i,"imageC"=>i+10_000))

    tacoRLEResult = alpha_taco_rle(B, C, 0.5)
    push!(results, Dict("kernel"=>kernel_str, "alpha"=>alpha,"kind"=>"taco_rle","time"=>tacoRLEResult,"dataset"=>"humansketches","imageB"=>i,"imageC"=>i+10_000))

    finchrepeat = alpha_finch(B, C, 0.5)
    push!(results, Dict("kernel"=>kernel_str, "alpha"=>alpha,"kind"=>"finch_repeat","time"=>finchrepeat,"dataset"=>"humansketches","imageB"=>i,"imageC"=>i+10_000))

    finchSparse = alpha_finch_sparse(B, C, 0.5)
    push!(results, Dict("kernel"=>kernel_str, "alpha"=>alpha,"kind"=>"finch_sparse","time"=>finchSparse,"dataset"=>"humansketches","imageB"=>i,"imageC"=>i+10_000))
    
end

open("alpha.json","w") do f
    JSON.print(f, results)
end