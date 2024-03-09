#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON
using SparseArrays
using Printf
using LinearAlgebra
using BenchmarkTools
using SparseArrays
using MatrixDepot
using Finch
using Graphs
using SimpleWeightedGraphs
using Base: summarysize

s = ArgParseSettings("Run graph experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "graphs_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "willow"
end

parsed_args = parse_args(ARGS, s)

include("datasets.jl")
include("shortest_paths.jl")
include("bfs.jl")

function bfs_finch_push_pull(mtx)
    A = pattern!(Tensor(SparseMatrixCSC(mtx)))
    AT = pattern!(Tensor(permutedims(SparseMatrixCSC(mtx))))
    time = @belapsed bfs_finch_kernel($A, $AT, 1)
    output = bfs_finch_kernel(A, AT, 1)
    return (; time = time, mem = summarysize(A), output = output)
end

function bfs_finch_push_only(mtx)
    A = pattern!(Tensor(SparseMatrixCSC(mtx)))
    AT = pattern!(Tensor(permutedims(SparseMatrixCSC(mtx))))
    time = @belapsed bfs_finch_kernel($A, $AT, 1, 0)
    output = bfs_finch_kernel(A, AT, 1, 0)
    return (; time = time, mem = summarysize(A), output = output)
end

function bfs_graphs(mtx)
    A = SimpleDiGraph(transpose(mtx))
    time = @belapsed Graphs.bfs_parents($A, 1)
    output = Graphs.bfs_parents(A, 1)
    return (; time = time, mem = summarysize(A), output = output)
end

function bellmanford_finch(mtx)
    A = redefault!(Tensor(Dense(SparseList(Element(0.0))), mtx), Inf)
    time = @belapsed bellmanford_finch_kernel($A, 1)
    output = bellmanford_finch_kernel(A, 1)
    return (; time = time, mem = summarysize(A), output = output)
end

function bellmanford_graphs(mtx)
    A = SimpleWeightedDiGraph(transpose(SparseMatrixCSC{Float64}(mtx)))
    time = @belapsed Graphs.bellman_ford_shortest_paths($A, 1)
    output = Graphs.bellman_ford_shortest_paths(A, 1)
    return (; time = time, mem = summarysize(A), output = output)
end

function check_bfs(A, src, res_parent, ref_parent)
    g = SimpleDiGraph(transpose(A))
    ref_levels = gdistances(g, src)
    for i in 1:nv(g)
        if ref_parent[i] == 0
            @assert res_parent[i] == 0
        elseif ref_parent[i] == i
            @assert res_parent[i] == i
        else
            @assert ref_levels[res_parent[i]] == ref_levels[i] - 1
        end
    end
    return true
end

function check_bellman(A, src, res, ref)
    n = length(ref.dists)
    for i in 1:n
        if ref.dists[i] != res.dists[i]
            @info "dists" i res.dists[i] ref.dists[i]
        end
        if ref.parents[i] != 0
            @assert A[res.parents[i], i] + ref.dists[res.parents[i]] == ref.dists[i]
        end
    end
    return true
end

results = []

for (op_name, check, methods) in [
    ("bfs",
        check_bfs,
        [
            "Graphs.jl" => bfs_graphs,
            "finch_push_pull" => bfs_finch_push_pull,
            "finch_push_only" => bfs_finch_push_only,
        ]
    ),
    ("bellmanford",
        check_bellman,
        [
            "Graphs.jl" => bellmanford_graphs,
            "Finch" => bellmanford_finch,
        ]
    ),
]
    for mtx in datasets[parsed_args["dataset"]]
        A = SparseMatrixCSC(matrixdepot(mtx))
        (n, n) = size(A)
        @info "testing" op_name mtx
        reference = nothing
        for (key, method) in methods
            display(A)
            result = method(A)

            time = result.time
            reference = something(reference, result.output)

            check(A, 1, result.output, reference) || @warn("incorrect result")

            # res.y == y_ref || @warn("incorrect result")
            @info "results" key result.time result.mem
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "operation" => op_name,
                "matrix" => mtx,
            ))
            write(parsed_args["output"], JSON.json(results, 4))
        end
    end
end