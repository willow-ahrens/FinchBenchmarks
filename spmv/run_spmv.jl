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

s = ArgParseSettings("Run SPMV experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "spmv_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "oski"
    "--save_results", "-s"
        arg_type = Bool
        help = "save conjugate gradient result to file"
        default = true
end

parsed_args = parse_args(ARGS, s)

datasets = Dict(
    "oski" => [
        "Boeing/ct20stif",
    ]
)

include("spmv_finch.jl")
include("spmv_julia.jl")

results = []

for mtx in datasets[parsed_args["dataset"]]
    A = SparseMatrixCSC(matrixdepot(mtx))
    (n, n) = size(A)
    x = rand(n)
    y = zeros(n)
    y_ref = nothing
    for (key, method) in [
        "julia" => spmv_julia,
        "finch" => spmv_finch
    ] 
        @info "testing" key mtx
        res = method(y, A, x)
        if parsed_args["save_results"]
            rm(key * "_results.txt", force=true)
            open(key * "_results.txt","a") do io
                for i = 1:n
                    @printf(io,"%f\n", res.y[1][i])
                end
            end
        end
        time = res.time
        y_ref = something(y_ref, res.y)

        n_ref = norm(y_ref)
        if key == "finch"
            t = 0.0
            for i = 1:n
                t += res.y[1][i]^2
            end
            n_res = sqrt(t)
        else
            n_res = norm(res.y)
        end
        diff = (n_res - n_ref) / n_ref 
        @info "difference" diff
        diff < 0.1 || @warn("incorrect result via norm")

        # res.y == y_ref || @warn("incorrect result")
        @info "results" time
        push!(results, OrderedDict(
            "time" => time,
            "method" => key,
            "kernel" => "spmv",
            "matrix" => mtx,
        ))
        write(parsed_args["output"], JSON.json(results, 4))
    end
end