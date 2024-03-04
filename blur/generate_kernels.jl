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
    for (input, output, tmp) in [
        [
            Tensor(Dense(Dense(Element(UInt(0))))),
            Tensor(Dense(Dense(Element(UInt(0))))),
            Tensor(Dense(Element(UInt(0))))
        ],
        [
            Tensor(Dense(SparseList(Element(UInt(0))))),
            Tensor(Dense(SparseList(Element(UInt(0))))),
            Tensor(SparseList(Element(UInt(0))))
        ],
        [
            Tensor(Dense(DenseRLE(Element(UInt(0))))),
            Tensor(Dense(DenseRLE(Element(UInt(0)), merge=false))),
            Tensor(DenseRLE(Element(UInt(0)), merge=false))
        ],
    ]
    push!(kernels, Finch.@finch_kernel function erode_finch_bits_kernel(output, input, tmp)
        output .= 0
        for y = _
            tmp .= 0
            for x = _
                tmp[x] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
            end
            for x = _
                let tl = coalesce(tmp[~(x-1)], ~(UInt(0))), t = tmp[x], tr = coalesce(tmp[~(x+1)], ~(UInt(0)))
                    output[x, y] = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
                end
            end
        end
        return output
    end)
    end

    for (input, output, tmp, mask) in [
        [
            Tensor(Dense(Dense(Element(UInt(0))))),
            Tensor(Dense(Dense(Element(UInt(0))))),
            Tensor(Dense(Element(UInt(0)))),
            Tensor(Dense(SparseList(Pattern())))
        ],
    ]
        push!(kernels, Finch.@finch_kernel function erode_finch_bits_mask_kernel(output, input, tmp, mask)
            output .= 0
            for y = _
                tmp .= 0
                for x = _
                    if mask[x, y]
                        tmp[x] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
                    end
                end
                for x = _
                    if mask[x, y]
                        let tl = coalesce(tmp[~(x-1)], ~(UInt(0))), t = tmp[x], tr = coalesce(tmp[~(x+1)], ~(UInt(0)))
                            output[x, y] = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
                        end
                    end
                end
            end
            return output
        end)
    end

    for (input, output, tmp) in [
        [
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Element(false))),
        ],
        [
            Tensor(Dense(SparseList(Pattern()))),
            Tensor(Dense(SparseList(Pattern()))),
            Tensor(SparseList(Pattern()))
        ],
        [
            Tensor(Dense(SparseRLE(Pattern())))
            Tensor(Dense(SparseRLE(Pattern(), merge=false)))
            Tensor(SparseRLE(Pattern(), merge=false))
        ],
    ]
        push!(kernels, Finch.@finch_kernel function erode_finch_kernel(output, input, tmp)
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
            return output
        end)
    end

    Serialization.serialize(kernels_file, kernels)
end

generate(joinpath(@__DIR__, "kernels.jls"))