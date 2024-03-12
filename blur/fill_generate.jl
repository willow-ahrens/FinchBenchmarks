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
    for (output, input, mask, tmp) in [
        [
            Tensor(Dense(Dense(Element(UInt(0))))),
            Tensor(Dense(Dense(Element(UInt(0))))),
            Tensor(Dense(Dense(Element(UInt(0))))),
            Tensor(Dense(Element(UInt(0))))
        ],
    ]
        push!(kernels, Finch.@finch_kernel function fill_finch_bits_kernel(output, input, mask, tmp)
            output .= 0
            for y = _
                tmp .= 0
                for x = _
                    tmp[x] = coalesce(input[x, ~(y-1)], UInt(0)) | input[x, y] | coalesce(input[x, ~(y+1)], UInt(0))
                end
                for x = _
                    let tl = coalesce(tmp[~(x-1)], UInt(0)), t = tmp[x], tr = coalesce(tmp[~(x+1)], UInt(0))
                        let t2 = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) | t | ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
                            output[x, y] = t2 & mask[x, y]
                        end
                    end
                end
            end
            return output
        end)
    end

    for (output, input, mask, tmp) in [
        [
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Element(false))),
        ],
        [
            Tensor(Dense(SparseRLE(Pattern(), merge=false)))
            Tensor(Dense(SparseRLE(Pattern())))
            Tensor(Dense(SparseRLE(Pattern())))
            Tensor(SparseRLE(Pattern(), merge=false))
        ],
        [
            Tensor(SparseList(SparseRLE(Pattern(), merge=false)))
            Tensor(SparseList(SparseRLE(Pattern())))
            Tensor(Dense(SparseRLE(Pattern())))
            Tensor(SparseRLE(Pattern(), merge=false))
        ],
    ]
        push!(kernels, Finch.@finch_kernel function fill_finch_kernel(output, input, mask, tmp)
            output .= false
            for y = _
                tmp .= false
                for x = _
                    tmp[x] = coalesce(input[x, ~(y-1)], false) | input[x, y] | coalesce(input[x, ~(y+1)], false)
                end
                for x = _
                    let t2 = coalesce(tmp[~(x-1)], false) | tmp[x] | coalesce(tmp[~(x+1)], false)
                        output[x, y] = mask & t2
                    end
                end
            end
            return output
        end)
    end

    Serialization.serialize(kernels_file, kernels)
end

generate(joinpath(@__DIR__, "fill_kernels.jls"))