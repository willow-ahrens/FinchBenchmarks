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
using LinearAlgebra
using Printf

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
    for (key, method) in [
        "iterative_solvers" => cg_iterative_solvers,
        "finch" => cg_finch
    ] 
        @info "testing" key mtx
        res = method(x, A, b, num_iters)

        # Uncomment to manually compare results
        #=
            rm(key * "_results.txt", force=true)
            open(key * "_results.txt","a") do io
                for i = 1:n
                    @printf(io,"%f\n", res.x[i])
                end
            end
        =#

        time = res.time
        x_ref = something(x_ref, res.x)
        @info "norm" norm(A * AsArray(res.x) - b)/norm(b)

        norm(res.x - x_ref)/norm(x_ref) < 0.1 || @warn("incorrect result via norm")

        # res.x == x_ref || @warn("incorrect result")
        @info "results" time
        push!(results, OrderedDict(
            "time" => time,
            "method" => key,
            "kernel" => "cg",
            "matrix" => mtx,
            "num_iters" => num_iters, 
        ))
        write(parsed_args["output"], JSON.json(results, 4))
    end
end
