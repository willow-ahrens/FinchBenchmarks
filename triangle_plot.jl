using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
using Statistics
#unicodeplots()
pyplot()

include("plot_labels.jl")

data = DataFrame(open("triangle_results.json", "r") do f
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
    ylabel = "Speedup Over TACO"
)

savefig(p, "triangle.png")