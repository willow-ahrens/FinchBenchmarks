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

s = ArgParseSettings("Run SPMV experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "spmv_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "vuduc_symmetric"
end

parsed_args = parse_args(ARGS, s)

datasets = Dict(
    "vuduc_symmetric" => [
        "Boeing/ct20stif",
        "Simon/olafu",
        "Boeing/bcsstk35",
        "Boeing/crystk02",
        "Boeing/crystk03",
        "Nasa/nasasrb",
        "Rothberg/3dtube",
        "Simon/raefsky4",
        "Mulvey/finan512",
        "Pothen/pwt",
        "Cote/vibrobox",
        "HB/saylr4",
        "Gupta/gupta1"
    ],
    "vuduc_unsymmetric" => [
        "Simon/raefsky3",
        "Simon/venkat01",
        "FIDAP/ex11",
        "Zitney/rdist1",
        "HB/orani678",
        "Goodwin/rim",
        "Hamm/memplus",
        "HB/gemat11",
        "Mallya/lhr10",
        "Grund/bayer02",
        "Grund/bayer10",
        "Brethour/coater2",
        "ATandT/onetone2",
        "Wang/wang4",
        "HB/lnsp3937",
        "HB/sherman5",
        "HB/sherman3",
        "Shyy/shyy161",
        "Wang/wang3",
    ],
    "willow_symmetric" => [
        "GHS_indef/exdata_1",
        # "Janna/Emilia_923",
        # "Janna/Geo_1438",
        "TAMU_SmartGridCenter/ACTIVSg70K"
    ],
    "willow_unsymmetric" => [
        "Goodwin/Goodwin_071",
        "Hamm/scircuit",
        # "LPnetlib/lpi_gran",
        "Norris/heart3",
        "Rajat/rajat26",
        "TSOPF/TSOPF_RS_b678_c1"
    ],
    "permutation" => [
        "permutation_synthetic"
    ], 
    "graph_symmetric" => [
        "SNAP/email-Enron", 
        "SNAP/as-735",
        "SNAP/Oregon-1",
        "Newman/as-22july06",
        "SNAP/loc-Brightkite",
        "SNAP/as-Skitter"
    ],
    "graph_unsymmetric" => [
        "SNAP/soc-Epinions1",
        "SNAP/wiki-Vote",
        "SNAP/email-EuAll",
        "SNAP/cit-HepPh",
        "SNAP/web-NotreDame",
        "SNAP/amazon0302",
        "SNAP/p2p-Gnutella08",
        "SNAP/email-Eu-core",
    ]
    "banded" => [
        "small_band_synthetic",
        "medium_band_synthetic",
        "large_band_synthetic",
        "SNAP/cit-HepTh"
    ]
)

include("synthetic.jl")
include("spmv_finch.jl")
include("spmv_finch_int8.jl")
include("spmv_finch_pattern.jl")
include("spmv_finch_pattern_unsym.jl")
include("spmv_finch_unsym.jl")
include("spmv_finch_vbl.jl")
include("spmv_finch_vbl_int8.jl")
include("spmv_finch_vbl_pattern.jl")
include("spmv_finch_vbl_unsym.jl")
include("spmv_finch_band.jl")
include("spmv_finch_band_unsym.jl")
include("spmv_finch_point.jl")
include("spmv_julia.jl")
include("spmv_taco.jl")
include("spmv_suite_sparse.jl")

dataset_tags = Dict(
    "vuduc_symmetric" => "symmetric",
    "vuduc_unsymmetric" => "unsymmetric",
    "willow_symmetric" => "symmetric",
    "willow_unsymmetric" => "unsymmetric",
    "permutation" => "permutation",
    "banded" => "banded",
    "graph_symmetric" => "symmetric_pattern",
    "graph_unsymmetric" => "unsymmetric_pattern",
)

methods = Dict(
    "symmetric" => [
        "julia" => spmv_julia,
        "finch" => spmv_finch,
        "finch_unsym" => spmv_finch_unsym,
        "finch_vbl" => spmv_finch_vbl,
        "finch_vbl_unsym" => spmv_finch_vbl_unsym,
        "finch_band" => spmv_finch_band,
        "finch_band_unsym" => spmv_finch_band_unsym,
        "taco" => spmv_taco,
        "suite_sparse" => spmv_suite_sparse,
    ],
    "unsymmetric" => [
        "julia" => spmv_julia,
        "finch_unsym" => spmv_finch_unsym,
        "finch_vbl_unsym" => spmv_finch_vbl_unsym,
        "finch_band_unsym" => spmv_finch_band_unsym,
        "taco" => spmv_taco,
        "suite_sparse" => spmv_suite_sparse,
    ],
    "symmetric_pattern" => [
        "julia" => spmv_julia,
        "finch_pattern" => spmv_finch_pattern,
        "taco" => spmv_taco,
        "suite_sparse" => spmv_suite_sparse,
    ],
    "unsymmetric_pattern" => [
        "julia" => spmv_julia,
        "finch_pattern_unsym" => spmv_finch_pattern_unsym,
        "taco" => spmv_taco,
        "suite_sparse" => spmv_suite_sparse,
    ],
    "symmetric_quantized" => [
        "julia" => spmv_julia,
        "finch_int8" => spmv_finch_int8,
        "finch_vbl_int8" => spmv_finch_vbl_int8,
        "taco" => spmv_taco,
        "suite_sparse" => spmv_suite_sparse,
    ],
    "permutation" => [
        "julia" => spmv_julia,
        "finch_point" => spmv_finch_point,
        "taco" => spmv_taco,
        "suite_sparse" => spmv_suite_sparse,
    ],
    "banded" => [
        "julia" => spmv_julia,
        "finch" => spmv_finch,
        "finch_band" => spmv_finch_band,
        "finch_band_unsym" => spmv_finch_band_unsym,
        "taco" => spmv_taco,
        "suite_sparse" => spmv_suite_sparse,
    ]
)

results = []

int(val) = mod(floor(Int, val), Int8)

for (dataset, mtxs) in datasets
    tag = dataset_tags[dataset]
    for mtx in mtxs
        if dataset == "permutation"
            A = SparseMatrixCSC(random_permutation_matrix(200000))
        elseif dataset == "banded"
            if mtx == "small_band_synthetic"
                A = SparseMatrixCSC(banded_matrix(10000, 5))
            elseif mtx == "medium_band_synthetic"
                A = SparseMatrixCSC(banded_matrix(10000, 30))
            elseif mtx == "large_band_synthetic"
                A = SparseMatrixCSC(banded_matrix(10000, 100))
            end
        else
            A = SparseMatrixCSC(matrixdepot(mtx))
        end

        # A = map((val) -> int(val), A)
        (n, n) = size(A)
        x = rand(n)
        # x = sprand(n, 0.1)
        y = zeros(n)
        y_ref = nothing
        for (key, method) in methods[tag]
            @info "testing" key mtx
            res = method(y, A, x)
            #=
                rm(key * "_results.txt", force=true)
                open(key * "_results.txt","a") do io
                    for i = 1:n
                        @printf(io,"%f\n", res.y[i])
                    end
                end
            =#
            time = res.time
            y_ref = something(y_ref, res.y)

            norm(res.y - y_ref)/norm(y_ref) < 0.1 || @warn("incorrect result via norm")

            # res.y == y_ref || @warn("incorrect result")
            @info "results" time
            push!(results, OrderedDict(
                "time" => time,
                "method" => key,
                "kernel" => "spmv",
                "matrix" => mtx,
                "dataset" => dataset,
            ))
            write(parsed_args["output"], JSON.json(results, 4))
        end
    end
end