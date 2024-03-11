#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using Finch
using Serialization

function generate(kernels_file)
    kernels = []
    for (img, mask, bins) in [
        [
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(Element(false)))
        ],
    ]
        push!(kernels, @finch_kernel function hist_finch_kernel(bins, img, mask)
            bins .= 0 
            for x=_
                for y=_
                    if mask[y, x]
                        bins[div(img[y, x], 16) + 1] += 1
                    end
                end
            end
            return bins
        end)
    end

    Serialization.serialize(kernels_file, kernels)
end

generate(joinpath(@__DIR__, "hist_kernels.jls"))