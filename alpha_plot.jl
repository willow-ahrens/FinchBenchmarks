using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
using Statistics
#unicodeplots()
pyplot()

include("plot_labels.jl")

data = DataFrame(open("alpha_results.json", "r") do f
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
    xlabel="Dataset",
    ylabel = "Speedup Over OpenCV"
)

savefig(p, "alpha.png")