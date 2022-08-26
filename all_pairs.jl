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

function all_pairs_finch_gallop_kernel(m, A, O)
    o = Scalar{0.0}()
    R = @fiber(d(e(0.0)))
    @finch @loop k ij R[k] += A[k, ij]^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += A[k, ij::gallop] * A[l, ij::gallop]))
end

function all_pairs_finch_gallop(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d(sl(e(0.0)))),A)
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_gallop_kernel($m, $A, $O)

    return finch_time, O
end

function all_pairs_finch_kernel(m, A, O)
    o = Scalar{0.0}()
    R = @fiber(d(e(0.0)))
    @finch @loop k ij R[k] += A[k, ij]^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += A[k, ij] * A[l, ij]))
end

function all_pairs_finch(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d(sl(e(0.0)))),A)
    O = fiber(zeros(Float64, num_imgs, num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $A, $O)

    return finch_time, O
end

function all_pairs_finch_vbl(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d(sv(e(0.0)))),A)
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $A, $O)

    return finch_time, O
end

function all_pairs_finch_rle(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = copyto!(@fiber(d(rl(0.0))),A)
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $A, $O)

    return finch_time, O
end

function all_pairs_finch_uint8_gallop_kernel(m, A, O)
    o = Scalar{0.0}()
    R = @fiber(d(e(0.0)))
    @finch @loop k ij R[k] += A[k, ij]^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += convert($(value(Float64)), A[k, ij::gallop]) * convert($(value(Float64)), A[l, ij::gallop])))
end

function all_pairs_finch_uint8_gallop(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d(sl(e(0.0)))),A)
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_gallop_kernel($m, $A, $O)

    return finch_uint8_time, O
end

function all_pairs_finch_uint8_kernel(m, A, O)
    o = Scalar{0.0}()
    R = @fiber(d(e(0.0)))
    @finch @loop k ij R[k] += A[k, ij]^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += convert($(value(Float64)), A[k, ij]) * convert($(value(Float64)), A[l, ij])))
end

function all_pairs_finch_uint8(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d(sl(e(0x00)))),A)
    O = fiber(zeros(Float64, num_imgs, num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_kernel($m, $A, $O)

    return finch_uint8_time, O
end

function all_pairs_finch_uint8_vbl(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d(sv(e(0x00)))),A)
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_kernel($m, $A, $O)

    return finch_uint8_time, O
end

function all_pairs_finch_uint8_rle(A, num_imgs)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = copyto!(@fiber(d(rl(0x00))),A)
    O = fiber(zeros(Float64,num_imgs,num_imgs))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d(sl(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_kernel($m, $A, $O)

    return finch_uint8_time, O
end


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

num_imgs = 40
datasets = []

function main(result_file)
    open(result_file,"w") do f
        println(f, "[")
    end

    for (mtx, key) in [
        ("mnist_train", "mnist"),
        ("emnist_letters_train","emnist_letters"),
        ("emnist_digits_train","emnist_digits"),
        ("omniglot_train", "omniglot"),
        ("cifar10_train", "cifar10"),
        ]

        println(key)
        push!(datasets, key)

        A = matrixdepot(mtx)
        if ndims(A) == 3
            A = A[:, :, randperm(end)]
        elseif ndims(A) == 4
            A = A[:, :, :, randperm(end)]
            A = (A .>> 4) .<< 4
            A = reshape(A, size(A, 1), size(A, 2), :)
        end

        opencv_time, result = all_pairs_opencv(A, num_imgs, key)
        println("opencv time: ", opencv_time)

        finch_time, result = all_pairs_finch(A, num_imgs)
        println("Finch time : ", finch_time, " -- ", opencv_time/finch_time, "x faster than OpenCV")

        finch_gallop_time, result = all_pairs_finch_gallop(A, num_imgs)
        println("Finch (gallop) time : ", finch_gallop_time, " -- ", opencv_time/finch_gallop_time, "x faster than OpenCV")

        finch_vbl_time, result = all_pairs_finch_vbl(A, num_imgs)
        println("Finch (vbl) time : ", finch_vbl_time, " -- ", opencv_time/finch_vbl_time, "x faster than OpenCV")

        finch_rle_time, result = all_pairs_finch_rle(A, num_imgs)
        println("Finch (rle) time : ", finch_rle_time, " -- ", opencv_time/finch_rle_time, "x faster than OpenCV")

        finch_uint8_time, result = all_pairs_finch_uint8(A, num_imgs)
        println("Finch time : ", finch_uint8_time, " -- ", opencv_time/finch_uint8_time, "x faster than OpenCV")

        finch_uint8_gallop_time, result = all_pairs_finch_uint8_gallop(A, num_imgs)
        println("Finch (gallop) time : ", finch_uint8_gallop_time, " -- ", opencv_time/finch_uint8_gallop_time, "x faster than OpenCV")

        finch_uint8_vbl_time, result = all_pairs_finch_uint8_vbl(A, num_imgs)
        println("Finch (vbl) time : ", finch_uint8_vbl_time, " -- ", opencv_time/finch_uint8_vbl_time, "x faster than OpenCV")

        finch_uint8_rle_time, result = all_pairs_finch_uint8_rle(A, num_imgs)
        println("Finch (rle) time : ", finch_uint8_rle_time, " -- ", opencv_time/finch_uint8_rle_time, "x faster than OpenCV")

        open(result_file,"a") do f
            println()
            JSON.print(f, Dict(
                "matrix"=>mtx,
                "n"=>size(A,1),
                "opencv_time"=>opencv_time,
                "finch_time"=>finch_time,
                "finch_gallop_time"=>finch_gallop_time,
                "finch_vbl_time"=>finch_vbl_time,
                "finch_rle_time"=>finch_rle_time,
                "finch_uint8_time"=>finch_uint8_time,
                "finch_uint8_gallop_time"=>finch_uint8_gallop_time,
                "finch_uint8_vbl_time"=>finch_uint8_vbl_time,
                "finch_uint8_rle_time"=>finch_uint8_rle_time,
            ))
            println(f, ",")
        end
    end

    open(result_file,"a") do f
        println(f, "]")
    end
end

main(ARGS...)
