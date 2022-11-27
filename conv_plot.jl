using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
using Statistics
#unicodeplots()
pyplot()

include("plot_labels.jl")

data = DataFrame(open("conv_results.json", "r") do f
    JSON.parse(f)
end)

interest = [
    "opencv",
    "finch_sparse",
]

p = plot(
    xlabel="Density",
    ylabel = "Runtime",
    xscale = :log,
    yscale = :log,
    xflip = true
)
for method in interest
    target = data[isequal(method).(data.method), :]
    plot!(p,
        target.p,
        target.time,
        label=label(method)
    )
end

savefig(p, "conv.png")