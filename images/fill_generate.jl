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

function generate(kernels_file)
    kernels = []
    for (frontier_2, frontier, mask, image, tmp) in [
        [
            Tensor(Dense(SparseList(Pattern()))),
            Tensor(Dense(SparseList(Pattern()))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(SparseList(Pattern()))),
        ],
        [
            Tensor(SparseList(SparseList(Pattern()))),
            Tensor(SparseList(SparseList(Pattern()))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(SparseList(SparseList(Pattern()))),
        ],
    ]
        push!(kernels, Finch.@finch_kernel function fill_finch_step_kernel(frontier_2, frontier, mask, image, tmp)
            tmp .= false
            for y = _
                for x = _
                    if image[x, y] && !mask[x, y]
                        tmp[x, y] = coalesce(frontier[x, ~(y-1)], false) || coalesce(frontier[x, ~(y+1)], false)
                    end
                end
            end
            frontier_2 .= false
            for y = _
                for x = _
                    if image[x, y] && !mask[x, y]
                        frontier_2[x, y] = coalesce(frontier[~(x-1), y], false) || tmp[x, y] || coalesce(frontier[~(x+1), y], false)
                    end
                end
            end
            for y = _
                for x = _
                    mask[x, y] |= frontier_2[x, y]
                end
            end
            return frontier_2
        end)
    end

    for (frontier_2, frontier, mask, image, c) in [
        [
            Tensor(Dense(SparseByteMap(Pattern()))),
            Tensor(Dense(SparseByteMap(Pattern()))),
            Tensor(Dense(Dense(Element(false)))),
            Tensor(Dense(Dense(Element(false)))),
            Scalar(0)
        ],
    ]
        push!(kernels, Finch.@finch_kernel function fill_finch_scatter_step_kernel(frontier_2, frontier, mask, image, c)
            frontier_2 .= false
            for y = _
                for x = _
                    if frontier[x, y]
                        if image[identity(x - 1), y] && !mask[identity(x - 1), y]
                            frontier_2[identity(x - 1), y] = true
                        end
                        if image[identity(x + 1), y] && !mask[identity(x + 1), y]
                            frontier_2[identity(x + 1), y] = true
                        end
                        if image[x, identity(y - 1)] && !mask[x, identity(y - 1)]
                            frontier_2[x, identity(y - 1)] = true
                        end
                        if image[x, identity(y + 1)] && !mask[x, identity(y + 1)]
                            frontier_2[x, identity(y + 1)] = true
                        end
                    end
                end
            end
            for y = _
                for x = _
                    let f = frontier_2[x, y]
                        mask[x, y] |= f
                        c[] += f
                    end
                end
            end
            return (frontier_2, c)
        end)
    end

    Serialization.serialize(kernels_file, kernels)
end

generate(joinpath(@__DIR__, "fill_kernels.jls"))