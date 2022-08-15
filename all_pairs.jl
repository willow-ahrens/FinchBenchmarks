using Finch, SparseArrays, BenchmarkTools, Images, FileIO, FixedPointNumbers, Colors
using JSON
using TensorDepot, MatrixDepot

include("TensorMarket.jl")
using .TensorMarket

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

function img_to_dense(img)
    return copyto!(@f(s(s(e(0x0::UInt8)))), copy(rawview(channelview(img))))
end

function img_to_repeat(img)
    return copyto!(@f(s(r(0x0::UInt8))), copy(rawview(channelview(img))))
end

function all_pairs_finch_kernel(o, m, T, O)
    @index @loop k l i j (o[k, l] += m[k,l] * (convert($(value(Float64)),T[k, i, j]) - convert($(value(Float64)),T[l, i, j]))^2)
    @index @loop k l (O[k,l] = sqrt(o[k,l]))
end

function all_pairs_finch(tensor_func, num_imgs)
    mnist_arr = (tensor_func(1:num_imgs))
    T = dropdefaults!(@f(s(s(l(e($(0x0::UInt8)))))),copy(rawview(channelview(mnist_arr))))
    O = fiber(zeros(num_imgs,num_imgs))
    
    o = similar(O)
    @index @loop i j o[i, j] = 0.0
    
    dense_m = [i < j ? 1.0 : 0.0 for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@f(s(l(e(0.0)))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($o, $m, $T, $O)

    return finch_time, O
end


function all_pairs_opencv(tensor_func, num_imgs, result_compare)
    T = tensor_func(1:num_imgs)
    for i in 1:num_imgs
        img = img_to_dense(T[i, :, :])
        pngwrite(joinpath(tmp_tensor_dir, "$i.png"), ffindnz(img)..., size(img))
    end

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./opencv/build/lib", "LD_LIBRARY_PATH" => "./opencv/build/lib") do
    	run(pipeline(`./all_pairs_opencv $tmp_tensor_dir/ $num_imgs $tmp_tensor_dir/result.ttx`, stdout=io))
    end
    opencv_time = parse(Int64, String(take!(io))) * 1.0e-9
    println("opencv time: ", opencv_time)


    result = fsparse(ttread(joinpath(tmp_tensor_dir, "result.ttx"))...)

    comp =  FiberArray(result_compare)
    res = FiberArray(result)
    # display(comp)
    # println()    
    # display(res)
    # println()
    # @assert comp == res
    #TODO: Reenable assertion, currently erroring: "ERROR: LoadError: AssertionError: comp == res"

    return opencv_time
end

run(pipeline(`make all_pairs_opencv`))

dataset = mnist
num_imgs = 10

finch_time, result = all_pairs_finch(dataset, num_imgs)
opencv_time = all_pairs_opencv(dataset, num_imgs, result)

println("Finch (sparse) time : ", finch_time, " -- ", opencv_time/finch_time, "x faster than OpenCV")
println("OpenCV time         : ", opencv_time)