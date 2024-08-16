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
        xflip = true,
        legendfontsize=12,
        xtickfontsize=16,
        ytickfontsize=16,
        size=(6 * 200, 3 * 200),
        xgrid=false,
        bar_width=0.4,
        dpi=200,
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