using Base: nothing_sentinel
#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end
include("../../deps/diagnostics.jl")
print_diagnostics()

using MatrixDepot
using BenchmarkTools
using ArgParse
using DataStructures
using JSON

# Parsing Arguments
s = ArgParseSettings("Run Parallel SpMV Experiments.")
@add_arg_table! s begin
    "--output", "-o"
    arg_type = String
    help = "output file path"
    "--dataset", "-d"
    arg_type = String
    help = "dataset keyword"
end
parsed_args = parse_args(ARGS, s)

# Mapping from dataset types to datasets
datasets = Dict(
    "uniform" => [
        OrderedDict("size" => 1000, "sparsity" => 0.1),
        OrderedDict("size" => 10000, "sparsity" => 0.1),
    ],
    "FEMLAB" => [
        "FEMLAB/poisson3Da",
        "FEMLAB/poisson3Db",
    ],
)

# Mapping from method keywords to methods
include("reference.jl")
include("parallel_row.jl")
methods = OrderedDict(
    "reference" => reference_mul,
    "parallel_row" => parallel_row_mul,
)

function calculate_results(dataset, mtxs, results)
    for mtx in mtxs
        # Get relevant matrix
        if dataset == "uniform"
            A = fsprand(mtx["size"], mtx["size"], mtx["sparsity"])
        elseif dataset == "FEMLAB"
            A = matrixdepot(mtx)
        else
            throw(ArgumentError("Cannot recognize dataset: $dataset"))
        end

        (num_rows, num_cols) = size(A)
        # x is a dense vector
        x = rand(num_rows)
        # y is the result vector
        y = zero(num_cols)

        for (key, method) in methods
            result = method(y, A, x)

            # Check the result of the multiplication
            reference_result = reference_mul(y, A, x)
            @assert result.y == reference_result.y "Incorrect result for $key"

            # Write result
            time = result.time
            @info "result for $key on $mtx" time
            push!(results, OrderedDict(
                "time" => time,
                "n_threads" => Threads.nthreads(),
                "method" => key,
                "dataset" => dataset,
                "matrix" => mtx,
            ))
            if isnothing(parsed_args["output"])
                write("results/spmv_$(Threads.nthreads())_threads.json", JSON.json(results, 4))
            else
                write(parsed_args["output"], JSON.json(results, 4))
            end
        end
    end
end

results = []
if isnothing(parsed_args["dataset"])
    for (dataset, mtxs) in datasets
        calculate_results(dataset, mtxs, results)
    end
else
    dataset = parsed_args["dataset"]
    mtxs = datasets[dataset]
    calculate_results(dataset, mtxs, results)
end


