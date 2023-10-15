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

#Here is where we use the julia arg parser to collect an input dataset keyword and an output file path

s = ArgParseSettings("Run conjugate gradient experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "cg_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "oski"
    "--num_iters"
        arg_type = Int
        help = "number of iters to run"
        default = 20
end

parsed_args = parse_args(ARGS, s)

num_iters = parsed_args["num_iters"]

datasets = Dict(
    "oski" => [
        "Boeing/ct20stif",
    ]
)

include("cg_finch.jl")
include("cg_iterative_solvers.jl")

results = []

for mtx in datasets[parsed_args["dataset"]]
    A = SparseMatrixCSC(matrixdepot(mtx))
    (n, n) = size(A)
    b = rand(n)
    x = zeros(n)
    x_ref = nothing
    for method in [
        (;
            key = "iterative_solvers",
            run = cg_iterative_solvers,
            format = cg_iterative_solvers_format
        ),
        (;
            key = "finch",
            run = cg_finch,
            format = cg_finch_format
        )
    ] 
        @info "testing" method.key mtx
        (_x, _A, _b) = method.format(x, A, b)
        res = Ref{Any}()
        time = @belapsed $res[] = $(method.run)($_x, $_A, $_b, $num_iters)
        x_ref = something(x_ref, res[])
        res[] == x_ref || @warn("incorrect result")
        @info "results" time
        push!(results, OrderedDict(
            "time" => time,
            "method" => method.key,
            "kernel" => "cg",
            "matrix" => mtx,
            "num_iters" => num_iters, 
        ))
        write(parsed_args["output"], JSON.json(results, 4))
    end
end
