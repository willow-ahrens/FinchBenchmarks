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
    return copyto!(@fiber(d(d(e(0x0::UInt8)))), copy(rawview(channelview(img))))
end

function img_to_repeat(img)
    return copyto!(@fiber(d(r(0x0::UInt8))), copy(rawview(channelview(img))))
end

@inline sq(x) = x * x

Finch.register()

function all_pairs_finch_gallop_kernel(m, T, O)
    o = Scalar{0.0}()
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(o[])) where (@loop ij o[] += $sq(convert($(value(Float64)),T[k, ij::gallop]) - convert($(value(Float64)),T[l, ij::gallop]))))
end

function all_pairs_finch_gallop(tensor_func, num_imgs)
    mnist_arr = (tensor_func(1:num_imgs))
    T = dropdefaults!(@fiber(d(sl(e($(0x00::UInt8))))),reshape(copy(rawview(channelview(mnist_arr))), num_imgs, :))
    println(length(copy(rawview(channelview(mnist_arr)))))
    println(length(T.lvl.lvl.lvl.val))
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_gallop_kernel($m, $T, $O)

    return finch_time, O
end

function all_pairs_finch_kernel(m, T, O)
    o = Scalar{0.0}()
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(o[])) where (@loop ij o[] += $sq(convert($(value(Float64)),T[k, ij]) - convert($(value(Float64)),T[l, ij]))))
end

function all_pairs_finch(tensor_func, num_imgs)
    mnist_arr = (tensor_func(1:num_imgs))
    T = dropdefaults!(@fiber(d(sl(e($(0x00::UInt8))))),reshape(copy(rawview(channelview(mnist_arr))), num_imgs, :))
    println(length(copy(rawview(channelview(mnist_arr)))))
    println(length(T.lvl.lvl.lvl.val))
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $T, $O)

    return finch_time, O
end

function all_pairs_finch_vbl(tensor_func, num_imgs)
    mnist_arr = (tensor_func(1:num_imgs))
    T = dropdefaults!(@fiber(d(sv(e($(0x00::UInt8))))),reshape(copy(rawview(channelview(mnist_arr))), num_imgs, :))
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $T, $O)

    return finch_time, O
end

function all_pairs_finch_rle(tensor_func, num_imgs)
    mnist_arr = (tensor_func(1:num_imgs))
    T = copyto!(@fiber(d(rl($(0x00::UInt8)))),reshape(copy(rawview(channelview(mnist_arr))), num_imgs, :))
    println(length(T.lvl.lvl.val))
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $T, $O)

    return finch_time, O
end


function all_pairs_opencv(tensor_func, num_imgs, result_compare)
    T = tensor_func(1:num_imgs)
    for i in 1:num_imgs
        img = img_to_dense(T[i, :, :])
        pngwrite(joinpath(tmp_tensor_dir, "$i.png"), ffindnz(img)..., size(img))
        pngwrite("$i.png", ffindnz(img)..., size(img))
    end

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./opencv/build/lib", "LD_LIBRARY_PATH" => "./opencv/build/lib") do
    	run(pipeline(`./all_pairs_opencv $tmp_tensor_dir/ $num_imgs $tmp_tensor_dir/result.ttx`, stdout=io))
    end
    opencv_time = parse(Int64, String(take!(io))) * 1.0e-9
    println("opencv time: ", opencv_time)


    result = fsparse(ttread(joinpath(tmp_tensor_dir, "result.ttx"))...)

    return (opencv_time, result)
end

run(pipeline(`make all_pairs_opencv`))


dataset = mnist
num_imgs = 20

opencv_time, result = all_pairs_opencv(dataset, num_imgs, result)

finch_time, result = all_pairs_finch(dataset, num_imgs)
println("Finch time : ", finch_time, " -- ", opencv_time/finch_time, "x faster than OpenCV")

finch_gallop_time, result = all_pairs_finch_gallop(dataset, num_imgs)
println("Finch (gallop) time : ", finch_gallop_time, " -- ", opencv_time/finch_gallop_time, "x faster than OpenCV")

finch_vbl_time, result = all_pairs_finch_vbl(dataset, num_imgs)
println("Finch (vbl) time : ", finch_vbl_time, " -- ", opencv_time/finch_vbl_time, "x faster than OpenCV")

finch_rle_time, result = all_pairs_finch_rle(dataset, num_imgs)
println("Finch (rle) time : ", finch_rle_time, " -- ", opencv_time/finch_rle_time, "x faster than OpenCV")