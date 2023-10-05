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
Plots.scalefontsizes(2.5)

include("plot_labels.jl")

function main(infile, outname)
    data = DataFrame(open(infile, "r") do f
        JSON.parse(f)
    end)

    interest = [
        "finch_sparse",
        "finch_gallop",
        "finch_lead",
        "finch_follow",
        "finch_vbl",
    ]

    ref = data[isequal("taco_sparse").(data.method), [:matrix, :x, :run, :time]]
    rename!(ref, :time => :ref)
    data = outerjoin(data, ref, on = [:matrix, :x, :run])
    data.speedup = data.ref ./ data.time

    data = data[map(m -> m in interest, data.method), :]

    target = data[isequal("0.1 density").(data.x), :]

    p = boxplot(
        CategoricalArray(label.(target.method), levels=label.(interest)),
        target.speedup,
        #xlabel="Method",
        xtickfontrotation=15,
        ylabel = "Speedup Over TACO",
        legend=false
    )
    hline!(p, [1], label=false)

    savefig(p, "$(outname)_10dense.png")

    target = data[isequal("10 count").(data.x), :]

    p = boxplot(
        CategoricalArray(label.(target.method), levels=label.(interest)),
        target.speedup,
        #xlabel="Method",
        xtickfontrotation=15,
        ylabel = "Speedup Over TACO",
        legend=false
    )
    hline!(p, [1], label=false)

    savefig(p, "$(outname)_10count.png")
end

main(ARGS...)
