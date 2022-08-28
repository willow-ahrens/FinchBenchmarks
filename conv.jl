using Finch, SparseArrays, BenchmarkTools, Images, FileIO, FixedPointNumbers, Colors
using JSON
using MatrixDepot,TensorDepot
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
    @finch @loop i j k l C[i, k] += (A[i, k] != 0) * coalesce(A[permit[offset[6-i, j]], permit[offset[6-k, l]]], 0) * coalesce(F[permit[j], permit[l]], 0)
end

function conv_finch_time(A, F)
    C = similar(A)
    @belapsed conv_finch_kernel($C, $A, $F)
end

#=
function all_pairs_opencv(A, num_imgs, key)
    persist_dir = joinpath(get_scratch!("Finch-CGO-2023"), "allpairs_opencv_$(key)")

    result_file = joinpath(mktempdir(prefix="allpairs_opencv_$(key)"), "result.ttx")

    for i in 1:num_imgs
        img = A[:, :, i]
        pngwrite(joinpath(persist_dir, "$i.png"), ffindnz(img)..., size(img))
    end

    io = IOBuffer()

    
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./opencv/build/lib", "LD_LIBRARY_PATH" => "./opencv/build/lib") do
    	run(pipeline(`./all_pairs_opencv $persist_dir/ $num_imgs $result_file`, stdout=io))
    end
    opencv_time = parse(Int64, String(take!(io))) * 1.0e-9

    result = fsparse(ttread(result_file)...)

    return (opencv_time, result)
end
=#

num_imgs = 256
datasets = []

function main(result_file)
    open(result_file,"w") do f
        println(f, "[")
    end

    for p in [0.1, 0.01, 0.001, 0.0001]

        A = pattern(fsprand((1000, 1000), p))
        F = rand(11, 11)

        open(result_file,"a") do f
            println()
            finch_time = conv_finch_time(A, F)
            println(finch_time)
            JSON.print(f, Dict(
                "matrix"=>mtx,
                "n"=>size(A,1),
                "finch_time"=>finch_time,
            ))
            println(f, ",")
        end
    end

    open(result_file,"a") do f
        println(f, "]")
    end
end

main(ARGS...)
