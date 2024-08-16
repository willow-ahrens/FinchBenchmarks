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
        "finch_sparse",
        "finch_gallop",
    ]

    ref = data[isequal("taco_sparse").(data.method), [:matrix, :time]]
    rename!(ref, :time => :ref)
    data = outerjoin(data, ref, on = [:matrix])
    data.speedup = data.ref ./ data.time

    data = data[map(m -> m in interest, data.method), :]
    group = CategoricalArray(label.(data.method), levels=label.(interest))

    p = boxplot(
        group,
        data.speedup,
        xlabel="Method",
        ylabel = "Speedup Over TACO",
        legend=false,
        legendfontsize=12,
        xtickfontsize=16,
        ytickfontsize=16,
        size=(6 * 200, 3 * 200),
        xgrid=false,
        bar_width=0.4,
        dpi=200,
    )
    hline!([1.0], line=:dash, color=:red, label=nothing)

    savefig(p, outfile)
end

main(ARGS...)