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

using LinearAlgebra
using ImageView

function main(resultfile)
    results = []

    for (dataset, getdata, I, f) in [
        ("testimage_dip3e", testimage_dip3e, remotefiles_dip3e, (img) -> Array{UInt8}(Array{Gray}(img) .> 0.1)),
    ]
        for i in I
            input = f(getdata(i))

            A = Tensor(SparseList(SparseRLE(Element(UInt8(0)))), input)

            if countstored(A)/prod(size(A)) <= 0.004
                Images.save("masks/$(splitext(i)[1]).png", Array{Gray}(input))
            end
        end
    end
end

main("test.json")
