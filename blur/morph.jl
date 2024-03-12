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

include("erode.jl")
include("hist.jl")

sobel(img) = abs.(imfilter(img, Kernel.sobel()[1])) + abs.(imfilter(img, Kernel.sobel()[2]))

using LinearAlgebra

MAG_FACTOR = 16
magnifying_lens = ones(UInt8, MAG_FACTOR, MAG_FACTOR)

function flip(img)
    if sum(img) > length(img) / 2
        return .~(img)
    else
        return img
    end
end

function main(resultfile)
    OpenCV.setNumThreads(1)

    results = []

    for (dataset, getdata, I, f) in [
        ("testimage_dip3e", testimage_dip3e, dip3e_masks[1:4], (img) -> Array{UInt8}(Array{Gray}(img) .> 0.1)),
        ("testimage_dip3e_magnify", testimage_dip3e, dip3e_masks[1:4], (img) -> kron(Array{UInt8}(Array{Gray}(img) .> 0.1), magnifying_lens)),
        ("humansketches", humansketches, 1:4, (img) -> Array{UInt8}(reinterpret(UInt8, img) .< 0xF0)),
        ("humansketches_magnify", humansketches, 1:4, (img) -> kron(Array{UInt8}(reinterpret(UInt8, img) .< 0xF0), magnifying_lens)),
        ("omniglot", omniglot_train, 1:4, (img) -> Array{UInt8}(img .!= 0x00)),
        ("omniglot_magnify", omniglot_train, 1:4, (img) -> kron(Array{UInt8}(img .!= 0x00), magnifying_lens)),
        ("mnist", mnist_train, 1:4, (img) -> Array{UInt8}(img .> 0x02)),
        ("mnist_magnify", mnist_train, 1:4, (img) -> kron(Array{UInt8}(img .> 0x02), magnifying_lens)),
        #("testimage_dip3e_edge", testimage_dip3e, ["FigP1039.tif"], (img) -> Array{UInt8}(sobel(Array{Gray}(img)) .> 0.1)),
        #("testimage_edge", testimage, ["airplaneF16.tiff", "fabio_color_512.png"], (img) -> Array{UInt8}(sobel(Array{Gray}(img)) .> 0.1)),
        #("willow", willow_gen, [800, 1600, 3200], identity),
        #("mnist_edge", mnist_train, 1:4, (img) -> Array{UInt8}(sobel(img) .> 90)),
        #("omniglot_edge", omniglot_train, 1:4, (img) -> Array{UInt8}(sobel(img) .> 0.1)),
        #("humansketches_edge", humansketches, 1:4, (img) -> Array{UInt8}(sobel(img) .> 0.1)),
    ]
        for i in I
            input = f(getdata(i))
            rand_data = rand(UInt8, size(input)...)

            for (op, kernels) in [
                #=
                ("histblur", [
                    (method = "opencv", fn = histblur_opencv(rand_data)),
                    (method = "finch", fn = histblur_finch(rand_data)),
                    (method = "finch_rle", fn = histblur_finch_rle(rand_data)),
                ]),
                ("blur", [
                    (method = "opencv", fn = blur_opencv(rand_data)),
                    (method = "finch", fn = blur_finch(rand_data)),
                    (method = "finch_rle", fn = blur_finch_rle(rand_data)),
                ]),
                =#
                ("hist", [
                    (method = "opencv", fn = hist_opencv(rand_data)),
                    (method = "finch", fn = hist_finch(rand_data)),
                    (method = "finch_rle", fn = hist_finch_rle(rand_data)),
                ]),
                #=
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
                =#
                #("dilate", [
                #    (method = "opencv", fn = dilate_opencv),
                #    (method = "finch_rle", fn = dilate_finch_rle),
                #    (method = "finch", fn = dilate_finch),
                #    (method = "finch_bits", fn = dilate_finch_bits),
                #])
            ]

                reference = nothing

                for kernel in kernels
                    result = kernel.fn(input)

                    reference = something(reference, result.output)
                    if reference != result.output
                        @info "oops" reference AsArray(result.output)
                    end
                    @assert reference == result.output

                    println("$op, $dataset [$i]: $(kernel.method) time: ", result.time, "\tmem: ", result.mem, "\tnnz: ", result.nnz)

                    push!(results, Dict("op" => op, "dataset"=>dataset, "i" => i, "method"=> kernel.method, "mem" => result.mem, "nnz" => result.nnz, "time"=>result.time))
                    write(resultfile, JSON.json(results, 4))
                end
            end
        end
    end

    return results
end

main("test.json")
