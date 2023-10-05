using Finch, SparseArrays, BenchmarkTools, Images, FileIO, FixedPointNumbers, Colors
using TensorDepot, MatrixDepot
using Finch.IndexNotation:literal_instance
using TensorMarket

const MyInt = Int32

using Scratch
tmp_tensor_dir = ""
if haskey(ENV, "TMP_TENSOR_DIR")
    tmp_tensor_dir = ENV["TMP_TENSOR_DIR"]
else
    tmp_tensor_dir = get_scratch!(@__MODULE__, "tmp_tensor_dir")
end

function pngwrite(filename, I, V, shape)
    @boundscheck begin
        length(shape) ⊆ 2:3 || error("Grayscale or RGB(A) only")
    end

    if length(shape) == 2
        out = Array{Gray{N0f8}, 2}(undef, shape[1],shape[2])

        for (coord, val) in zip(zip(I...), V)
            out[coord[1], coord[2]] = reinterpret(N0f8, convert(UInt8,val))
        end

        save(filename, out)
    else 
        if shape[3] == 3
            out = Array{RGB{N0f8}, 2}(undef, shape[1],shape[2])
            out_raw = rawview(channelview(out))
            for (coord, val) in zip(zip(I...), V)
                out_raw[coord[3], coord[1], coord[2]] = reinterpret(N0f8, convert(UInt8,val))
            end
            save(filename, out)
        elseif shape[4] == 4
            out = Array{RGBA{N0f8}, 2}(undef, shape[1],shape[2])
            out_raw = rawview(channelview(out))
            for (coord, val) in zip(zip(I...), V)
                out_raw[coord[3], coord[1], coord[2]] = reinterpret(N0f8, convert(UInt8,val))
            end
            save(filename, out)
        else 
            error("Array must be RGB or RGBA")
        end
    end
end

function img_to_dense(img)
    return copyto!(@fiber(d{MyInt}(d{MyInt}(e(0x0::UInt8)))), copy(rawview(channelview(img))))
end

function img_to_repeat(img)
    return copyto!(@fiber(d{MyInt}(rl{0x0::UInt8, MyInt, MyInt}())), copy(rawview(channelview(img))))
end

function img_to_repeat_diffs(img)
    return copyto!(@fiber(d{MyInt}(rld{0x0::UInt8, MyInt, MyInt}())), copy(rawview(channelview(img))))
end

function alpha_opencv(B, C, alpha)
    APath = joinpath(tmp_tensor_dir, "A.png")
    ARefPath = joinpath(tmp_tensor_dir, "A_ref.png")
    BPath = joinpath(tmp_tensor_dir, "B.png")
    CPath = joinpath(tmp_tensor_dir, "C.png")

    as = Scalar{0.0, Float32}(alpha)
    mas = Scalar{0.0, Float32}(1- alpha)
    Bf = img_to_dense(B)
    Cf = img_to_dense(C)
    A_ref = img_to_dense(B)
    
    @finch @loop i j A_ref[i, j] = 0

    pngwrite(APath, ffindnz(A_ref)..., size(A_ref))
    pngwrite(BPath, ffindnz(Bf)..., size(Bf))
    pngwrite(CPath, ffindnz(Cf)..., size(Cf))

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./opencv/build/lib", "LD_LIBRARY_PATH" => "./opencv/build/lib") do
    	run(pipeline(`./alpha_opencv $APath $BPath $CPath $alpha`, stdout=io))
    end

    return (parse(Int64, String(take!(io))) * 1.0e-9, rawview(channelview(load(APath))))
end

function writeRLETacoTTX(filename, src)
    sz = size(src)
    rows = []
    cols = []
    vals = Vector{UInt8}()
    for i in 1:sz[1]
        curr = UInt8(src[i,1])
        push!(rows, i)
        push!(cols, 1)
        push!(vals, curr)
        for j in 1:sz[2]
            if src[i,j] != curr
                curr = UInt8(src[i,j])
                push!(rows, i)
                push!(cols, j)
                push!(vals, curr)
            end
        end
    end
    ttwrite(filename, (rows,cols), vals, size(src))
end

function alpha_taco_rle(B, C, alpha)
    APath = joinpath(tmp_tensor_dir, "A.ttx")
    ADensePath = joinpath(tmp_tensor_dir, "A_dense.ttx")
    BPath = joinpath(tmp_tensor_dir, "B.ttx")
    CPath = joinpath(tmp_tensor_dir, "C.ttx")
   
    as = Scalar{0.0, Float64}(alpha)
    mas = Scalar{0.0, Float64}(1- alpha)

    Bf = img_to_repeat(B)
    # Cf = img_to_repeat(C)

    writeRLETacoTTX(APath, zeros(UInt8, size(Bf)))
    writeRLETacoTTX(BPath, copy(rawview(channelview(B))))
    writeRLETacoTTX(CPath, copy(rawview(channelview(C))))

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./taco-rle/build/lib", "LD_LIBRARY_PATH" => "./taco-rle/build/lib", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        run(pipeline(`./alpha_taco_rle $APath $BPath $CPath $alpha $ADensePath`, stdout=io))
    end

    return (parse(Int64, String(take!(io))) * 1.0e-9, fsparse(ttread(ADensePath)...))
end

#@inline function unsafe_round_UInt8(x)
#    unsafe_trunc(UInt8, round(x))
#end
#
#Finch.register()

function alpha_finch_kernel(A, B, C, as, mas)
    @finch @loop i j A[i, j] = unsafe_trunc(UInt8, round($(literal_instance(as)) * B[i, j] + $(literal_instance(mas)) * C[i, j]))
end

function alpha_finch_rle(B, C, alpha)
    as = alpha
    mas = 1 - alpha

    B = img_to_repeat(B)
    C = img_to_repeat(C)
    A = similar(B)
    time = @belapsed alpha_finch_kernel($A, $B, $C, $as, $mas) evals=1
    return (time, A)
end

function alpha_finch_rled(B, C, alpha)
    as = alpha
    mas = 1 - alpha

    B = img_to_repeat_diffs(B)
    C = img_to_repeat_diffs(C)
    A = similar(B)
    time = @belapsed alpha_finch_kernel($A, $B, $C, $as, $mas) evals=1
    return (time, A)
end

function alpha_finch_sparse(B, C, alpha)
    as = alpha
    mas = 1 - alpha

    B = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e($(0xff::UInt8))))), copy(rawview(channelview(B))))
    C = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e($(0xff::UInt8))))), copy(rawview(channelview(C))))

    A = similar(B)

    time = @belapsed alpha_finch_kernel($A, $B, $C, $as, $mas) evals=1
    return (time, A)
end

function main(result_file)
    numSketches = 10
    humansketchesA = matrixdepot("humansketches", 1:numSketches)
    humansketchesB = matrixdepot("humansketches", (10_001):(10_000+numSketches))

    open(result_file, "w") do f
        println(f, "[")
    end

    comma = false

    for (humansketchesA, humansketchesB, key) in [
        (matrixdepot("humansketches", 1:numSketches), matrixdepot("humansketches", (10_001):(10_000+numSketches)), "humansketches"),
        (permutedims(matrixdepot("omniglot_train")[:, :, 1:numSketches], (3, 1, 2)), permutedims(matrixdepot("omniglot_train")[:, :, 10_001:10_000+numSketches], (3, 1, 2)), "omniglot_train"),
    ]
        for i in 1:numSketches
            B = humansketchesA[i, :, :]
            C = humansketchesB[i, :, :]

            time, reference = alpha_opencv(B, C, 0.5)

            for (method, timer) in [
                ("opencv", alpha_opencv),
                ("taco_rle", alpha_taco_rle),
                ("finch_rle", alpha_finch_rle),
                ("finch_rled", alpha_finch_rled),
                ("finch_sparse", alpha_finch_sparse)
            ]
                time, result = timer(B, C, 0.5)
                check = Scalar(true)
                @finch @loop i j check[] &= reference[i, j] == result[i, j]
                @assert check[]
                open(result_file, "a") do f
                    if comma
                        println(f, ",")
                    end
                    print(f, """
                        {
                            "dataset": $(repr(key)),
                            "imageB": $i,
                            "imageC": $(i+10_000),
                            "method": $(repr(method)),
                            "time": $time
                        }""")
                end
                @info "alpha" method key i time

                comma = true
            end
        end
    end

    open(result_file,"a") do f
        println(f)
        println(f, "]")
    end
end

main(ARGS...)
