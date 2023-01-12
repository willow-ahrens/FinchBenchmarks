using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
#unicodeplots()
pyplot()
default(size=(1600,600))
default(dpi=300)
Plots.scalefontsizes(3.0)

include("plot_labels.jl")

function main(infile, outfile)
    data = DataFrame(open(infile, "r") do f
        JSON.parse(f)
    end)

    interest = [
        "finch_sparse",
        "finch_gallop",
        "finch_vbl",
        "finch_rle",
    ]

    ref = data[isequal("opencv").(data.method), [:matrix, :time]]
    rename!(ref, :time => :ref)
    data = outerjoin(data, ref, on = [:matrix])
    data.speedup = data.ref ./ data.time

    data = data[map(m -> m in interest, data.method), :]
    group = CategoricalArray(label.(data.method), levels=label.(interest))
    matrix = CategoricalArray(label.(data.matrix), levels=label.(["mnist_train", "emnist_letters_train", "emnist_digits_train", "omniglot_train"]))

    p = groupedbar(matrix,
        data.speedup,
        group=group,
        legend=:topleft,
        #xlabel="Dataset",
        ylabel = "Speedup Over OpenCV"
    )

    hline!(p, [1], label=false)

    savefig(p, outfile)
end

main(ARGS...)