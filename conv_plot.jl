using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
using Statistics
#unicodeplots()
pyplot()

include("plot_labels.jl")

function main(infile, outfile)
    data = DataFrame(open(infile, "r") do f
        JSON.parse(f)
    end)

    interest = [
        "opencv",
        "finch_sparse",
    ]

    p = plot(
        xlabel="Sparsity (% Nonzero)",
        ylabel = "Runtime (s)",
        xscale = :log,
        yscale = :log,
        xflip = true
    )
    for method in interest
        target = data[isequal(method).(data.method), :]
        plot!(p .* 100,
            target.p,
            target.time,
            label=label(method)
        )
    end

    savefig(p, outfile)
end

main(ARGS...)