using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
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

    p = groupedbar(data.matrix,
        data.speedup,
        group=group,
        legend=:topleft,
        xlabel="Dataset",
        ylabel = "Speedup Over OpenCV",
        linecolor=nothing,
        legendfontsize=12,
        xtickfontsize=16,
        ytickfontsize=16,
        size=(6 * 200, 3 * 200),
        xgrid=false,
        bar_width=0.6,
        dpi=200,
    )

    savefig(p, outfile)
end

main(ARGS...)