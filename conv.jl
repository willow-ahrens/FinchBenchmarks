using Finch, SparseArrays, BenchmarkTools, Images, FileIO, FixedPointNumbers, Colors
using JSON
using Scratch
using Random

include("TensorMarket.jl")
using .TensorMarket


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
            out = Array{RGB{N0f8}, 2}(0x0, shape[1],shape[2])
            out_raw = rawview(channelview(out))
            for (coord, val) in zip(zip(I...), V)
                out_raw[coord[3], coord[1], coord[2]] = reinterpret(N0f8, convert(UInt8,val))
            end
            save(filename, out)
        elseif shape[4] == 4
            out = Array{RGBA{N0f8}, 2}(RGBA(), shape[1],shape[2])
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

function conv_finch_kernel(C, A, F)
    @finch @loop i k j l C[i, k] += (A[i, k] != 0) * coalesce(A[permit[offset[6-i, j]], permit[offset[6-k, l]]::fastwalk], 0) * coalesce(F[permit[j], permit[l]], 0)
end

function conv_finch_time(A, F)
    C = similar(A)
    A = pattern!(A)
    F = copyto!(@fiber(d(d(e(0.0)))), F)
    time = @belapsed conv_finch_kernel($C, $A, $F)
    @finch @loop i k j l C[i, k] += (A[i, k] != 0) * coalesce(A[permit[offset[6-i, j]], permit[offset[6-k, l]]::fastwalk], 0) * coalesce(F[permit[j], permit[l]], 0)
    return (time, C)
end

function conv_dense_kernel(C, A, F)
    (m, n) = size(A)
    C .= 0
    for k = 1:n
        for l = 1:11
            if 1 <= k-6+l <= n
                for i = 1:m
                    if A[i, k] != 0
                        for j = 1:11
                            if 1 <= i-6+j <= m
                                C[i, k] += A[i-6+j, k-6+l] * F[j, l]
                            end
                        end
                    end
                end
            end
        end
    end
    return C
end

function conv_dense_time(A, F)
    (m, n) = size(FiberArray(A))
    A = copyto!(Array{UInt8}(undef, m, n), FiberArray(A))
    C = Array{UInt8}(undef, m, n)
    time = @belapsed conv_dense_kernel($C, $A, $F)
    return (time, C)
end

function all_pairs_opencv(A, num_imgs, key)
    A_file = joinpath(mktempdir(prefix="conv_opencv_$(key)"), "A.png")
    C_file = joinpath(mktempdir(prefix="conv_opencv_$(key)"), "C.ttx")

    pngwrite(A_file, ffindnz(A_file)..., size(img))

    io = IOBuffer()
    
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./opencv/build/lib", "LD_LIBRARY_PATH" => "./opencv/build/lib") do
    	run(pipeline(`./all_pairs_opencv $persist_dir/ $num_imgs $result_file`, stdout=io))
    end
    opencv_time = parse(Int64, String(take!(io))) * 1.0e-9

    C = fsparse(ttread(C_file)...)

    return (opencv_time, C)
end

num_imgs = 256
datasets = []

function main(result_file)
    open(result_file,"w") do f
        println(f, "[")
    end

    for p in [0.1, 0.01, 0.001, 0.0001]

        A = copyto!(@fiber(d(sl(e(0x00)))), pattern!(fsprand((1000, 1000), p)))
        F = ones(UInt8, 11, 11)

        open(result_file,"a") do f
            println()
            finch_time, finch_C = conv_finch_time(A, F)
            println("finch", finch_time)
            dense_time, dense_C = conv_dense_time(A, F)
            println("dense", dense_time)
            #display(Int.(dense_C))
            #display(Int.(copyto!(similar(dense_C), FiberArray(finch_C))))
            @assert dense_C == FiberArray(finch_C)
            JSON.print(f, Dict(
                "p"=>p,
                "finch_time"=>finch_time,
                "dense_time"=>dense_time,
            ))
            println(f, ",")
        end
    end

    open(result_file,"a") do f
        println(f, "]")
    end
end

main(ARGS...)
