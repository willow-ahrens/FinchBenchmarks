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
    for (bins, img, mask) in [
        [
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(Dense(Element(false))))
        ],
        [
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(SparseRLE(Pattern())))
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

    #=
    for (output, input, tmp, mask) in [
        [
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Element(UInt(0)))),
            Tensor(Dense(SparseRLE(Pattern()))),
        ],
        [
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Dense(Element(UInt8(0))))),
            Tensor(Dense(Element(UInt(0)))),
            Tensor(Dense(Dense(Element(false)))),
        ],
    ]
        push!(kernels, Finch.@finch_kernel function blur_finch_kernel(output, input, tmp, mask)
            output .= false
            for y = _
                tmp .= false
                for x = _
                    if coalesce(mask[~(x - 1), y], false) || mask[x, y] || coalesce(mask[~(x + 1), y], false)
                        tmp[x] = UInt(coalesce(input[x, ~(y-1)], 0)) + UInt(input[x, y]) + UInt(coalesce(input[x, ~(y+1)], 0))
                    end
                end
                for x = _
                    if mask[x, y]
                        output[x, y] = unsafe_trunc(UInt8, round((UInt(coalesce(tmp[~(x-1)], 0)) + tmp[x] + UInt(coalesce(tmp[~(x+1)], 0)))/9))
                    end
                end
            end
            return output
        end)
    end

    for (bins, img, tmp, mask) in [
        [
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(false))))
        ],
        [
            Tensor(Dense(Element(0)))
            Tensor(Dense(Dense(Element(UInt8(0)))))
            Tensor(Dense(Element(0)))
            Tensor(Dense(SparseRLE(Pattern())))
        ],
    ]
        push!(kernels, @finch_kernel function histblur_finch_kernel(bins, img, tmp, mask)
            bins .= 0 
            for y = _
                tmp .= false
                for x = _
                    tmp[x] = UInt(coalesce(img[x, ~(y-1)], 0)) + UInt(img[x, y]) + UInt(coalesce(img[x, ~(y+1)], 0))
                end
                for x = _
                    if mask[x, y]
                        let t = unsafe_trunc(UInt8, round((UInt(coalesce(tmp[~(x-1)], 0)) + tmp[x] + UInt(coalesce(tmp[~(x+1)], 0)))/9))
                            bins[div(t, 16) + 1] += 1
                        end
                    end
                end
            end
            return bins
        end)
    end
    =#

    Serialization.serialize(kernels_file, kernels)
end

generate(joinpath(@__DIR__, "hist_kernels.jls"))