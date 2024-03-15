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
using Finch

#Here is where we use the julia arg parser to collect an input dataset keyword and an output file path

s = ArgParseSettings("Run spgemm experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "spgemm_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        #default = "joel_sm"
        default = "joel_lg"
    "--num_iters"
        arg_type = Int
        help = "number of iters to run"
        default = 20
end

parsed_args = parse_args(ARGS, s)

num_iters = parsed_args["num_iters"]

datasets = Dict(
    "short" => [
        "HB/bcspwr07",
    ],
    "joel_sm" => [
        "FEMLAB/poisson3Da",
        "SNAP/wiki-Vote",
    ],
    "joel_lg" => [
        "SNAP/email-Enron",
        "SNAP/ca-CondMat",
        "Oberwolfach/filter3D",
        "Williams/cop20k_A",
        "Um/offshore",
        "Um/2cubes_sphere",
        "vanHeukelum/cage12",
        "SNAP/amazon0312",
        "Hamm/scircuit",
        "SNAP/web-Google",
        "GHS_indef/mario002",
        "SNAP/cit-Patents",
        "JGD_Homology/m133-b3",
        "Williams/webbase-1M",
        "SNAP/roadNet-CA",
        "SNAP/p2p-Gnutella31",
        "Pajek/patents_main"
    ]
)

include("spgemm_finch.jl")
include("spgemm_taco.jl")

function norm_tensor(C_ref, C)
        diff_val = Scalar(0.0)
        ref_val = Scalar(0.0)
        @finch begin
                diff_val .= 0
                for i=_,j=_
                        diff_val[] += (C_ref[j,i] - C[j,i]) * (C_ref[j,i] - C[j,i])
                end
        end
        @finch begin
                ref_val .= 0
                for i=_,j=_
                        ref_val[] += C_ref[j,i] * C_ref[j,i]
                end
        end
        return (sqrt(diff_val[])/sqrt(ref_val[]))

end

results = []

function norm_tensor(C_ref, C)
        diff_val = Scalar(0.0)
        ref_val = Scalar(0.0)
        @finch begin
                diff_val .= 0
                for i=_,j=_
                        diff_val[] += (C_ref[j,i] - C[j,i]) * (C_ref[j,i] - C[j,i])
                end
        end
        @finch begin
                ref_val .= 0
                for i=_,j=_
                        ref_val[] += C_ref[j,i] * C_ref[j,i]
                end
        end
        return (sqrt(diff_val[])/sqrt(ref_val[]))

end

for mtx in datasets[parsed_args["dataset"]]
    A = SparseMatrixCSC(matrixdepot(mtx))
    B = A
    C_ref = nothing
    for (key, method) in [
        "spgemm_taco_inner" => spgemm_taco_inner,
        "spgemm_taco_gustavson" => spgemm_taco_gustavson,
        #"spgemm_taco_outer" => spgemm_taco_outer,
        "spgemm_finch_inner" => spgemm_finch_inner,
        "spgemm_finch_gustavson" => spgemm_finch_gustavson,
        "spgemm_finch_outer" => spgemm_finch_outer,
    ]
    if parsed_args["dataset"] != "joel_lg" || key == "spgemm_taco_gustavson" || key == "spgemm_finch_gustavson"
        @info "testing" key mtx
        res = method(A, B)
        C_ref = something(C_ref, res.C)
        norm_tensor(C_ref, res.C) < 0.1 || @warn("incorrect result via norm")
        @info "results" res.time
        push!(results, OrderedDict(
            "time" => res.time,
            "method" => key,
            "kernel" => "spgemm",
            "matrix" => mtx,
        ))
        write(parsed_args["output"], JSON.json(results, 4))
    end
    end
end
