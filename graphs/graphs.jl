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
include("pagerank.jl")
include("shortest_paths.jl")
include("bfs.jl")
include("triangle_counting.jl")

function pagerank_finch(mtx)
    A = pattern!(Tensor(SparseMatrixCSC(mtx)))
    time = @belapsed pagerank_finch_kernel($A, nsteps=20, damp = 0.85)
    output = pagerank_finch_kernel(A)
    return (; time = time, mem = summarysize(A), output = Array(output))
end

function pagerank_graphs(mtx)
    A = Graphs.SimpleDiGraph(transpose(mtx))
    time = @belapsed Graphs.pagerank($A, 0.85, 20)
    output = Graphs.pagerank(A, 0.85, 20)
    return (; time = time, mem = summarysize(A), output = output)
end

function bfs_finch(mtx)
    A = pattern!(Tensor(SparseMatrixCSC(mtx)))
    time = @belapsed bfs_finch_kernel($A, 1)
    output = bfs_finch_kernel(A, 1)
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
    return (; time = time, mem = summarysize(A), output = collect(zip(output.dists, output.parents)))
end

results = []

for mtx in datasets[parsed_args["dataset"]]
    A = SparseMatrixCSC(matrixdepot(mtx))
    (n, n) = size(A)
    for (op_name, check, methods) in [
        ("pagerank",
            (x, y) -> norm(x - y)/norm(y) < 0.1,
            [
                "Graphs.jl" => pagerank_graphs,
                "Finch" => pagerank_finch,
            ]
        ),
        ("bfs",
            (==),
            [
                "Graphs.jl" => bfs_graphs,
                "Finch" => bfs_finch,
            ]
        ),
        ("bellmanford",
            (==),
            [
                "Graphs.jl" => bellmanford_graphs,
                "Finch" => bellmanford_finch,
            ]
        ),
    ]
        @info "testing" op_name mtx
        reference = nothing
        for (key, method) in methods
            result = method(A)

            time = result.time
            reference = something(reference, result.output)

            println(reference[1:5])
            println(result.output[1:5])

            check(reference, result.output) || @warn("incorrect result")

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