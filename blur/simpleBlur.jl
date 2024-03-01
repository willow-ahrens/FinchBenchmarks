#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using Finch
using TestImages
using ImageCore, OpenCV, TestImages, MosaicViews, Colors
using BenchmarkTools
using LinearAlgebra
using JSON
using Base: summarysize

blur_opencv_kernel(data) = OpenCV.blur(data, OpenCV.Size(Int32(3), Int32(3)))

function blur_opencv(input)
    time = @belapsed blur_opencv_kernel($input) evals=1
    return (; time = time, mem = summarysize(input), output = blur_opencv_kernel(input))
end

input = Tensor(Dense(Dense(Dense(Element(Float64(0))))))
output = Tensor(Dense(Dense(Dense(Element(Float64(0))))))
tmp = Tensor(Dense(Dense(Element(Float64(0)))))

eval(Finch.@finch_kernel function blur_finch_kernel(output, input, tmp)
    output .= 0
    for y = _
        tmp .= 0
        for x = _
            for c = _
                tmp[c, x] += (coalesce(input[c, ~(x-1), y], 0) + coalesce(input[c, x, y],0) + coalesce(input[c, ~(x+1), y], 0))/3
            end
        end
        for x = _
            for c = _
                output[c, x, y] += (coalesce(tmp[c, ~(x-1)],0) + coalesce(tmp[c, x], 0) + coalesce(tmp[c, ~(x+1)],0))/3
            end
        end
    end
end)

function blur_finch(img)
    (cs, xs, ys) = size(img)
    input = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img)
    output = Tensor(Dense(Dense(Dense(Element(Float64(0))))), undef, cs, xs, ys)
    tmp = Tensor(Dense(Dense(Element(Float64(0)))), undef, cs, xs)
    time = @belapsed blur_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), output=output)
end

input = Tensor(Dense(Dense(DenseRLE(Element(Float64(0))))))
output = Tensor(Dense(Dense(DenseRLE(Element(Float64(0)), merge=false))))
tmp = Tensor(Dense(DenseRLE(Element(Float64(0)), merge=false)))

eval(Finch.@finch_kernel function blur_finch_rle_kernel(output, input, tmp)
    output .= 0
    for x = _
        tmp .= 0
        for c = _
            for y = _
                tmp[y, c] += (coalesce(input[y, c, ~(x-1)], 0) + coalesce(input[y, c, x],0) + coalesce(input[y, c, ~(x+1)], 0))/3
            end
        end
        for c = _
            for y = _
                output[y, c, x] += (coalesce(tmp[~(y-1), c], 0) + coalesce(tmp[y, c], 0) + coalesce(tmp[~(y+1), c],0))/3
            end
        end
    end
end)

function blur_finch_rle(img)
    img = permutedims(img, (3, 1, 2))
    (ys, cs, xs) = size(img)
    input = Tensor(Dense(Dense(DenseRLE(Element(Float64(0))))), img)
    output = Tensor(Dense(Dense(DenseRLE(Element(Float64(0)), merge=false))), ys, cs, xs)
    tmp = Tensor(Dense(DenseRLE(Element(Float64(0)), merge=false)), ys, cs)
    time = @belapsed blur_finch_rle_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), output=permutedims(output, invperm((3, 1, 2))))
end

function testCorrect(img1, img2)
    img2AsDense = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img2)
    img1AsDense = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img1)
    return img2AsDense == img1AsDense
end

function runBlurRLE(input, output, tmp)
    blurRLE(input, output, tmp)
end

function main(resultfile)
    results = []

    for (filename, T) in [
        ("mandrill.tiff", Float64)
    ]

        data = testimage(filename)
        data_raw = Array{Float64}((channelview(data)))

        reference = nothing

        for kernel in [
            (method = "opencv", fn = blur_opencv),
            (method = "finch", fn = blur_finch),
            (method = "finch_rle", fn = blur_finch_rle),
        ]

            result = kernel.fn(data_raw)

            reference = something(reference, result.output)
            println(norm(reference .- result.output))
            #@assert reference == result.output

            println("$(kernel.method) time: ", result.time, "mem: ", result.mem)

            push!(results, Dict("imagename"=>filename, "method"=> kernel.method, "mem" => result.mem, "time"=>result.time))
            write(resultfile, JSON.json(results, 4))
        end
    end

    return results
end

main("test.json")
