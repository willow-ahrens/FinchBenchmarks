#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
include("../deps/diagnostics.jl")
print_diagnostics()

using Finch
using Serialization
#using TestImages
using OpenCV#, TestImages, MosaicViews, Colors, Images, FileIO
using BenchmarkTools
using LinearAlgebra
using ArgParse
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
include("fill.jl")

sobel(img) = abs.(imfilter(img, Kernel.sobel()[1])) + abs.(imfilter(img, Kernel.sobel()[2]))

using LinearAlgebra
using Random

function flip(img)
    if sum(img) > length(img) / 2
        return Array{UInt8}(img .== 0)
    else
        return img
    end
end

function prep_fill(img)
    (m, n) = size(img)
    pos = [(x, y) for x in 1:m for y in 1:n if img[x, y] != 0x00]
    (x, y) = pos[rand(1:end)]
    #if img[x, y] == 0
    #    img = Array{UInt8}(img .== 0x00)
    #end
    (img, x, y)
end

MAG_FACTOR = 8
magnifying_lens = ones(UInt8, MAG_FACTOR, MAG_FACTOR)

function magnify(img, factor)
    (m, n) = size(img)
    img = Array{UInt8}(reshape(img, 1, m, n))
    img = OpenCV.resize(img, OpenCV.Size{Int32}(m * factor, n * factor), interpolation = OpenCV.INTER_CUBIC)
    img = reshape(img, m * factor, n * factor)
end

s = ArgParseSettings("Run graph experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "morphology_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "all"
    "--num_trials", "-t"
        arg_type = Int
        help = "how many images to run"
        default = 4
end

parsed_args = parse_args(ARGS, s)
println(parsed_args)

OpenCV.setNumThreads(1)

results = []
N = parsed_args["num_trials"]

sample(r) = r[randperm(end)[1:min(N, end)]]

mnist_i = sample(1:mnist_train_length())
omniglot_i = sample(1:omniglot_train_length())
humansketches_i = sample(1:humansketches_length())
#testimage_dip3e_i = sample(dip3e_masks)
testimage_dip3e_i = sample(remotefiles_dip3e)

datas = Dict(
    "mnist" => (mnist_train, mnist_i, (img) -> Array{UInt8}(img .> 0x02)),
    "omniglot" => (omniglot_train, omniglot_i, (img) -> Array{UInt8}(img .== 0x00)),
    "humansketches" => (humansketches, humansketches_i, (img) -> Array{UInt8}(reinterpret(UInt8, img) .< 0xF0)),
    "testimage_dip3e" => (testimage_dip3e, testimage_dip3e_i, (img) -> Array{UInt8}(Array{Gray}(img) .> 0.1)),
    "mnist_magnify" => (mnist_train, mnist_i, (img) -> Array{UInt8}(magnify(img, MAG_FACTOR) .> 0x02)),
    "omniglot_magnify" => (omniglot_train, omniglot_i, (img) -> Array{UInt8}(magnify(img, MAG_FACTOR) .== 0x00)),
    "humansketches_magnify" => (humansketches, humansketches_i, (img) -> Array{UInt8}(magnify(reinterpret(UInt8, img), MAG_FACTOR) .< 0xF0)),
    "testimage_dip3e_magnify" => (testimage_dip3e, testimage_dip3e_i, (img) -> Array{UInt8}(magnify(Array{UInt8}(img*255), MAG_FACTOR) .> 0.1)),
)

groups = Dict(
    "all" => ["mnist", "omniglot", "humansketches", "testimage_dip3e", "mnist_magnify", "omniglot_magnify", "humansketches_magnify", "testimage_dip3e_magnify"],
    "standard" => ["mnist", "omniglot", "humansketches", "testimage_dip3e"],
    "magnify" => ["mnist_magnify", "omniglot_magnify", "humansketches_magnify", "testimage_dip3e_magnify"],
    "mnist" => ["mnist"],
    "omniglot" => ["omniglot"],
    "humansketches" => ["humansketches"],
    "testimage_dip3e" => ["testimage_dip3e"],
    "mnist_magnify" => ["mnist_magnify"],
    "omniglot_magnify" => ["omniglot_magnify"],
    "humansketches_magnify" => ["humansketches_magnify"],
    "testimage_dip3e_magnify" => ["testimage_dip3e_magnify"],
)

for dataset in groups[parsed_args["dataset"]]
    (getdata, I, f) = datas[dataset]
    for i in I
        input = f(getdata(i))

        for (op, prep, kernels) in [
            #("fill", prep_fill, [
            #    (method = "opencv", fn = fill_opencv),
            #    (method = "finch", fn = fill_finch),
            #    (method = "finch_scatter", fn = fill_finch_scatter),
            #]),
            ("erode2", (img) -> (img, 2), [
                (method = "opencv", fn = erode_opencv),
                (method = "finch", fn = erode_finch),
                (method = "finch_rle", fn = erode_finch_rle),
                (method = "finch_bits", fn = erode_finch_bits),
                (method = "finch_bits_mask", fn = erode_finch_bits_mask),
            ]),
            ("erode4", (img) -> (img, 4), [
                (method = "opencv", fn = erode_opencv),
                (method = "finch", fn = erode_finch),
                (method = "finch_rle", fn = erode_finch_rle),
                (method = "finch_bits", fn = erode_finch_bits),
                (method = "finch_bits_mask", fn = erode_finch_bits_mask),
            ]),
            ("hist", (img) -> (rand(UInt8, size(input)...), img), [
                (method = "opencv", fn = hist_opencv),
                (method = "finch", fn = hist_finch),
                (method = "finch_rle", fn = hist_finch_rle),
            ]),
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
        ]
            input2 = prep(input)

            reference = nothing

            for kernel in kernels
                result = kernel.fn(input2)

                #if reference == nothing && op in ["fill", "erode4"]
                #    Images.save("output/$(dataset)_$(op)_$(i).png", Array{Gray}(Array(result.output)))
                #end
                reference = something(reference, result.output)
                if reference != result.output
                    Images.save("output/$(dataset)_$(op)_$(i)_$(kernel.method).png", Array{Gray}(Array(result.output)))
                end
                @assert reference == result.output

                println("$op, $dataset [$i]: $(kernel.method) time: ", result.time, "\tmem: ", result.mem, "\tnnz: ", result.nnz)

                push!(results, Dict("operation" => op, "dataset"=>dataset, "label" => i, "method"=> kernel.method, "mem" => result.mem, "nnz" => result.nnz, "time"=>result.time))
                write(parsed_args["output"], JSON.json(results, 4))
            end
        end
    end
end
