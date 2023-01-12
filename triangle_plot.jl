using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
using Statistics
#unicodeplots()
pyplot()
default(size=(800,600))
default(dpi=300)
Plots.scalefontsizes(2.0)

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
        #xlabel="Method",
        ylabel = "Speedup Over TACO",
        legend=false
    )
    hline!(p, [1])

    savefig(p, outfile)
end

main(ARGS...)