using Finch, SparseArrays, BenchmarkTools, Images, FileIO, FixedPointNumbers, Colors
using MatrixDepot, TensorDepot
using Scratch
using Random
using TensorMarket

const MyInt = Int32

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
    R = @fiber(d{MyInt}(e(0.0)))
    @finch @loop k ij R[k] += A[k, ij]^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += A[k, ij::gallop] * A[l, ij::gallop]))
end

function all_pairs_finch_gallop(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_gallop_kernel($m, $A, $O) evals=1

    return finch_time, O
end

function all_pairs_finch_kernel(m, A, O)
    o = Scalar{0.0}()
    R = @fiber(d{MyInt}(e(0.0)))
    @finch @loop k ij R[k] += A[k, ij]^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += A[k, ij] * A[l, ij]))
end

function all_pairs_finch(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $A, $O) evals=1

    return finch_time, O
end

function all_pairs_finch_vbl(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d{MyInt}(sv{MyInt, MyInt}(e(0.0)))),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $A, $O) evals=1

    return finch_time, O
end

function all_pairs_finch_rle(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = copyto!(@fiber(d{MyInt}(rl{0.0, MyInt, MyInt}())), A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $A, $O) evals=1

    return finch_time, O
end

function all_pairs_finch_rled(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = copyto!(@fiber(d{MyInt}(rld{0.0, MyInt, MyInt}())), A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_time = @belapsed all_pairs_finch_kernel($m, $A, $O) evals=1

    return finch_time, O
end

function all_pairs_finch_uint8_gallop_kernel(m, A, O)
    o = Scalar{0.0}()
    R = @fiber(d{MyInt}(e(0.0)))
    @finch @loop k ij R[k] += convert(Float64, A[k, ij])^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += convert(Float64, A[k, ij::gallop]) * convert(Float64, A[l, ij::gallop])))
end

function all_pairs_finch_uint8_gallop(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0x00)))),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_gallop_kernel($m, $A, $O) evals=1

    return finch_uint8_time, O
end

function all_pairs_finch_uint8_kernel(m, A, O)
    o = Scalar{0.0}()
    R = @fiber(d{MyInt}(e(0.0)))
    @finch @loop k ij R[k] += convert(Float64, A[k, ij])^2
    @finch @loop k l @sieve m[k,l] ((O[k,l] = sqrt(R[k] + R[l] - 2 * o[])) where (@loop ij o[] += convert(Float64, A[k, ij]) * convert(Float64, A[l, ij])))
end

function all_pairs_finch_uint8(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0x00)))),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_kernel($m, $A, $O) evals=1

    return finch_uint8_time, O
end

function all_pairs_finch_uint8_vbl(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = dropdefaults!(@fiber(d{MyInt}(sv{MyInt, MyInt}(e(0x00)))),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_kernel($m, $A, $O) evals=1

    return finch_uint8_time, O
end

function all_pairs_finch_uint8_rle(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = copyto!(@fiber(d{MyInt}(rl{0x00, MyInt, MyInt}())),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_kernel($m, $A, $O) evals=1

    return finch_uint8_time, O
end

function all_pairs_finch_uint8_rled(A, num_imgs, key)
    A = reshape(permutedims(A[:, :, 1:num_imgs], (3, 1, 2)), num_imgs, :)
    A = copyto!(@fiber(d{MyInt}(rld{0x00, MyInt, MyInt}())),A)
    O = @fiber(d{MyInt}(num_imgs, d{MyInt}(num_imgs, e(0.0))))
    
    dense_m = [i < j for i in 1:num_imgs, j in 1:num_imgs]
    m = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(p()))), dense_m)

    finch_uint8_time = @belapsed all_pairs_finch_uint8_kernel($m, $A, $O) evals=1

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

num_imgs = 256

function main(result_file)
    comma = false

    open(result_file,"w") do f
        println(f, "[")
    end

    for (mtx, key) in [
        ("mnist_train", "mnist"),
        ("emnist_letters_train","emnist_letters"),
        ("emnist_digits_train","emnist_digits"),
        ("omniglot_train", "omniglot"),
    ]

        A = matrixdepot(mtx)
        if ndims(A) == 3
            A = A[:, :, randperm(end)]
        elseif ndims(A) == 4
            A = A[:, :, :, randperm(end)]
            A = (A .>> 4) .<< 4
            A = reshape(A, size(A, 1), size(A, 2), :)
        end

        (opencv_time, reference) = all_pairs_opencv(A, num_imgs, key)

        for (method, timer) = [
            "opencv"=>all_pairs_opencv,
            "finch_sparse"=>all_pairs_finch,
            "finch_gallop"=>all_pairs_finch_gallop,
            "finch_vbl"=>all_pairs_finch_vbl,
            "finch_rle"=>all_pairs_finch_rle,
            "finch_rled"=>all_pairs_finch_rled,
            "finch_uint8"=>all_pairs_finch_uint8,
            "finch_uint8_gallop"=>all_pairs_finch_uint8_gallop,
            "finch_uint8_vbl"=>all_pairs_finch_uint8_vbl,
            "finch_uint8_rle"=>all_pairs_finch_uint8_rle,
            "finch_uint8_rled"=>all_pairs_finch_uint8_rled,
        ]
            time, result = timer(A, num_imgs, key)

            check = Scalar(true)
            @finch @loop i j check[] &= abs(result[i, j] - reference[i, j]) < 0.1 
            #foo = Scalar(0.0)
            #@finch @loop i j foo[] <<max>>= abs(result[i, j] - reference[i, j])
            #println(foo)
            @assert check[]
            open(result_file,"a") do f
                if comma
                    println(f, ",")
                end
                print(f, """
                    {
                        "matrix": $(repr(mtx)),
                        "n": $(size(A, 1)),
                        "method": $(repr(method)),
                        "time": $time
                    }""")
            end
            @info "all pairs" mtx size(A, 1) method time
            comma = true
        end
    end

    open(result_file,"a") do f
        println(f)
        println(f, "]")
    end
end

main(ARGS...)
