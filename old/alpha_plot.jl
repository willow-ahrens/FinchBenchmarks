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
        "taco_rle",
        "finch_sparse",
        "finch_rle",
    ]

    ref = data[isequal("opencv").(data.method), [:dataset, :imageB, :imageC, :time]]
    rename!(ref, :time => :ref)
    data = outerjoin(data, ref, on = [:dataset, :imageB, :imageC])
    data.speedup = data.ref ./ data.time

    data = combine(groupby(data, [:method, :dataset]), :speedup=>mean=>:speedup)

    data = data[map(m -> m in interest, data.method), :]
    group = CategoricalArray(label.(data.method), levels=label.(interest))
    dataset = CategoricalArray(label.(data.dataset), levels=label.(["omniglot_train", "humansketches"]))

    p = groupedbar(dataset,
        data.speedup,
        group=group,
#        xlabel="Dataset",
        ylabel = "Speedup Over OpenCV",
        legend = :topleft
    )

    hline!(p, [1], label=false)

    savefig(p, outfile)
end

main(ARGS...)