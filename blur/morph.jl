#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using MLDatasets
using Finch
#using TestImages
using OpenCV#, TestImages, MosaicViews, Colors, Images, FileIO
using BenchmarkTools
using LinearAlgebra
using JSON
using Base: summarysize

download_cache = joinpath(@__DIR__, "../cache")

"""
mnist_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from mnist.
"""
function mnist_train()
    dir = joinpath(download_cache, "mnist")
    MNIST(:train, dir=dir, Tx=UInt8).features
end

"""
fashionmnist_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from fashionmnist.
"""
function fashionmnist_train()
    dir = joinpath(download_cache, "fashionmnist")
    FashionMNIST(:train, dir=dir, Tx=UInt8).features
end

"""
omniglot_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from omniglot.
"""
function omniglot_train()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:train, dir=dir, Tx=UInt8).features
end

"""
mnist_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from mnist.
"""
function mnist_test()
    dir = joinpath(download_cache, "mnist")
    MNIST(:test, dir=dir, Tx=UInt8).features
end

"""
fashionmnist_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from fashionmnist.
"""
function fashionmnist_test()
    dir = joinpath(download_cache, "fashionmnist")
    FashionMNIST(:test, dir=dir, Tx=UInt8).features
end

"""
omniglot_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from omniglot.
"""
function omniglot_test()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:test, dir=dir, Tx=UInt8).features
end

"""
omniglot_small1 dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the small1 split from omniglot.
"""
function omniglot_small1()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:small1, dir=dir, Tx=UInt8).features
end

"""
omniglot_small2 dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the small2 split from omniglot.
"""
function omniglot_small2()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:small2, dir=dir, Tx=UInt8).features
end

"""
emnist_digits_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from the digits extension of emnist.
"""
function emnist_digits_test()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:digits, :test, dir=dir, Tx=UInt8).features
end

"""
emnist_digits_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from the digits extentsion of mnist.
"""
function emnist_digits_train()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:digits, :train, dir=dir, Tx=UInt8).features
end

"""
emnist_letters_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from the letters extension of emnist.
"""
function emnist_letters_test()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:letters, dir=dir, Tx=UInt8).features
end

"""
emnist_letters_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from the letters extentsion of mnist.
"""
function emnist_letters_train()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:letters, :train, dir=dir, Tx=UInt8).features
end

"""
emnist_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from the complete emnist.
"""
function emnist_test()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:byclass, :test, dir=dir, Tx=UInt8).features
end

"""
emnist_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from complete emnist.
"""
function emnist_train()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:byclass, :train, dir=dir, Tx=UInt8).features
end

erode_opencv_kernel(data, filter) = OpenCV.erode(data, filter)

function erode_opencv(img)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed erode_opencv_kernel($input, $filter) evals=1
    output = dropdims(Array(erode_opencv_kernel(input, filter)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

input = Tensor(Dense(Dense(Element(false))))
output = Tensor(Dense(Dense(Element(false))))
tmp = Tensor(Dense(Element(false)))

eval(Finch.@finch_kernel function erode_finch_kernel(output, input, tmp)
    output .= false
    for y = _
        tmp .= false
        for x = _
            tmp[x] = coalesce(input[x, ~(y-1)], true) & input[x, y] & coalesce(input[x, ~(y+1)], true)
        end
        for x = _
            output[x, y] = coalesce(tmp[~(x-1)], true) & tmp[x] & coalesce(tmp[~(x+1)], true)
        end
    end
end)

function erode_finch(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

input = Tensor(Dense(SparseRLE(Pattern())))
output = Tensor(Dense(SparseRLE(Pattern(), merge=false)))
tmp = Tensor(SparseRLE(Pattern(), merge=false))

eval(Finch.@finch_kernel function erode_finch_rle_kernel(output, input, tmp)
    output .= false
    for y = _
        tmp .= false
        for x = _
            tmp[x, y] = coalesce(input[x, ~(y-1)], true) & input[x, y] & coalesce(input[x, ~(y+1)], true)
        end
        for x = _
            output[x, y] = coalesce(tmp[~(x-1), y], true) & tmp[x, y] & coalesce(tmp[~(x+1), y], true)
        end
    end
end)

function erode_finch_rle(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseRLE(Pattern(), merge=false)), undef, xs, ys)
    tmp = Tensor(SparseRLE(Pattern(), merge=false), undef, xs, ys)
    time = @belapsed erode_finch_rle_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function main(resultfile)
    results = []

    for (dataset, getdata, I, f) in [
        ("mnist", mnist_train, 1:1, (img) -> Array{UInt8}(img .> 0x02))
        ("omniglot", omniglot_train, 1:1, (img) -> Array{UInt8}(img .> 0x00))
    ]
        data = getdata()
        for i in I
            input = f(data[:, :, i])

            reference = nothing

            for kernel in [
                (method = "opencv", fn = erode_opencv),
                (method = "finch", fn = erode_finch),
                (method = "finch_rle", fn = erode_finch_rle),
            ]

                result = kernel.fn(input)

                reference = something(reference, result.output)
                @assert reference == result.output

                println("$(kernel.method) time: ", result.time, "\tmem: ", result.mem, "\tnnz: ", result.nnz)

                push!(results, Dict("imagename"=>"$dataset[$i]", "method"=> kernel.method, "mem" => result.mem, "nnz" => result.nnz, "time"=>result.time))
                write(resultfile, JSON.json(results, 4))
            end
        end
    end

    return results
end

main("test.json")
