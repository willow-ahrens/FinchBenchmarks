using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
using Statistics
#unicodeplots()
pyplot()
default(size=(800,500))
default(dpi=300)
Plots.scalefontsizes(2.5)

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
        xticks = 10.0 .^ (1:-1:-2),
        yticks = 10.0 .^ (-5:-1),
        xflip = true,
        legend = :bottomleft,
    )
    for method in interest
        target = data[isequal(method).(data.method), :]
        plot!(p,
            target.p .* 100,
            target.time,
            label=label(method)
        )
    end

    savefig(p, outfile)
end

main(ARGS...)