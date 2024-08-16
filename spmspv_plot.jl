using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
using Statistics
#unicodeplots()
pyplot()

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
        permute=(:x, :y)
    )
    vline!([1.0], line=:dash, color=:red, label=nothing)

    savefig(p, "$(outname)_1density.png")

    target = data[isequal("10 count").(data.x), :]

    p = boxplot(
        CategoricalArray(label.(target.method), levels=label.(interest)),
        target.speedup,
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
        permute=(:x, :y)
    )
    vline!([1.0], line=:dash, color=:red, label=nothing)

    savefig(p, "$(outname)_10count.png")
end

main(ARGS...)