#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using Finch
using Serialization
#using TestImages
using OpenCV#, TestImages, MosaicViews, Colors, Images, FileIO
using BenchmarkTools
using LinearAlgebra
using JSON
using Base: summarysize

include(joinpath(@__DIR__, "datasets.jl"))
for kernel in Serialization.deserialize(joinpath(@__DIR__, "kernels.jls"))
    eval(kernel)
end

erode_opencv_kernel(data, filter) = OpenCV.erode(data, filter)

function erode_opencv(img)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed erode_opencv_kernel($input, $filter) evals=1
    output = dropdims(Array(erode_opencv_kernel(input, filter)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

dilate_opencv_kernel(data, filter) = OpenCV.dilate(data, filter)

function dilate_opencv(img)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed dilate_opencv_kernel($input, $filter) evals=1
    output = dropdims(Array(dilate_opencv_kernel(input, filter)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

function erode_finch(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function dilate_finch(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed dilate_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function pack_bits(img)
    xs, ys = size(img)
    xb = cld(xs + 1, 64)
    imgb = fill(UInt(0), xb, ys)
    for y in 1:ys
        for x in 1:xs
            imgb[fld1(x, 64), y] |= UInt(Bool(img[x, y])) << (mod1(x, 64) - 1)
        end
    end
    imgb
end

function unpack_bits(imgb, xs, ys)
    img = zeros(UInt8, xs, ys)
    for y in 1:ys
        for x in 1:xs
            img[x, y] = UInt8((imgb[fld1(x, 64), y] >> (mod1(x, 64) - 1)) & 0x01)
        end
    end
    img
end

function erode_finch_bits(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function dilate_finch_bits(img)
    (xs, ys) = size(img)
    imgb = pack_bits(img .!= 0x00)
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed dilate_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_bits_sparse(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(SparseList(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(SparseList(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(SparseList(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_bits_rle(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(DenseRLE(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(DenseRLE(Element(UInt(0)), merge=false)), undef, xb, ys)
    tmpb = Tensor(DenseRLE(Element(UInt(0)), merge=false), undef, xb)
    time = @belapsed erode_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    outputb = erode_finch_bits_kernel(outputb, inputb, tmpb).output
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_bits_mask(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    maskb = Tensor(Dense(SparseList(Pattern())), imgb .!= 0)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_mask_kernel($outputb, $inputb, $tmpb, $maskb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_rle(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseRLE(Pattern(), merge=false)), undef, xs, ys)
    tmp = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function erode_finch_sparse(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseList(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseList(Pattern())), undef, xs, ys)
    tmp = Tensor(SparseList(Pattern()), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

sobel(img) = abs.(imfilter(img, Kernel.sobel()[1])) + abs.(imfilter(img, Kernel.sobel()[2]))

function main(resultfile)
    results = []

    for (dataset, getdata, I, f) in [
        ("testimage_dip3e", testimage_dip3e, ["Fig0534(a)(ellipse_and_circle).tif", "Fig0539(a)(vertical_rectangle).tif"], (img) -> Array{UInt8}(Array{Gray}(img) .> 0.1)),
        ("testimage_dip3e_edge", testimage_dip3e, ["FigP1039.tif"], (img) -> Array{UInt8}(sobel(Array{Gray}(img)) .> 0.1)),
        #("mnist", mnist_train, 1:4, (img) -> Array{UInt8}(img .> 0x02))
        ("testimage_edge", testimage, ["airplaneF16.tiff", "fabio_color_512.png"], (img) -> Array{UInt8}(sobel(Array{Gray}(img)) .> 0.1)),
        ("willow", willow_gen, [800, 1600, 3200], identity),
        ("humansketches", humansketches, 1:4, (img) -> Array{UInt8}(reinterpret(UInt8, img) .< 0xF0)),
        ("omniglot", omniglot_train, 1:4, (img) -> Array{UInt8}(img .!= 0x00)),
        #("mnist_edge", mnist_train, 1:4, (img) -> Array{UInt8}(sobel(img) .> 90)),
        #("mnist_edge", mnist_train, 1:4, (img) -> Array{UInt8}(sobel(img) .> 90)),
        ("omniglot_edge", omniglot_train, 1:4, (img) -> Array{UInt8}(sobel(img) .> 0.1)),
        ("humansketches_edge", humansketches, 1:4, (img) -> Array{UInt8}(sobel(img) .> 0.1)),
    ]
        for i in I
            input = f(getdata(i))

            for (op, kernels) in [
                ("erode", [
                    (method = "opencv", fn = erode_opencv),
                    (method = "finch", fn = erode_finch),
                    (method = "finch_rle", fn = erode_finch_rle),
                    #(method = "finch_sparse", fn = erode_finch_sparse),
                    (method = "finch_bits", fn = erode_finch_bits),
                    #(method = "finch_bits_sparse", fn = erode_finch_bits_sparse),
                    (method = "finch_bits_mask", fn = erode_finch_bits_mask),
                    #(method = "finch_bits_rle", fn = erode_finch_bits_rle),
                ]),
                ("dilate", [
                    (method = "opencv", fn = dilate_opencv),
                    #(method = "finch", fn = dilate_finch),
                    (method = "finch_bits", fn = dilate_finch_bits),
                ])
            ]

                reference = nothing

                for kernel in kernels
                    result = kernel.fn(input)

                    reference = something(reference, result.output)
                    @assert reference == result.output

                    println("$op, $dataset [$i]: $(kernel.method) time: ", result.time, "\tmem: ", result.mem, "\tnnz: ", result.nnz)

                    push!(results, Dict("op" => op, "imagename"=>"$dataset[$i]", "method"=> kernel.method, "mem" => result.mem, "nnz" => result.nnz, "time"=>result.time))
                    write(resultfile, JSON.json(results, 4))
                end
            end
        end
    end

    return results
end

main("test.json")
