using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
using CategoricalArrays
#unicodeplots()
pyplot()

data = DataFrame(open("all_pairs_results.json", "r") do f
    JSON.parse(f)
end)

interest = [
    "finch",
    "finch_gallop",
    "finch_vbl",
    "finch_rle",
]

labels = Dict(
    "mnist_train" => "MNIST",
    "emnist_letters_train" => "EMNIST Letters",
    "emnist_digits_train" => "EMNIST Digits",
    "omniglot_train" => "Omniglot",
    "opencv"=>"opencv",
    "finch"=>"Finch (Sparse)",
    "finch_gallop"=>"Finch (Gallop)",
    "finch_vbl"=>"Finch (VBL)",
    "finch_rle"=>"Finch (RLE)",
    "finch_uint8"=>"Finch (Sparse) (UInt8)",
    "finch_uint8_gallop"=>"Finch (Gallop) (UInt8)",
    "finch_uint8_vbl"=>"Finch (VBL) (UInt8)",
    "finch_uint8_rle"=>"Finch (RLE) (UInt8)"
)
data.matrix = map(key -> labels[key], data.matrix)

ref = data[isequal("opencv").(data.method), [:matrix, :time]]
rename!(ref, :time => :ref)
data = outerjoin(data, ref, on = [:matrix])
data.speedup = data.ref ./ data.time

data = data[map(m -> m in interest, data.method), :]
group = CategoricalArray(map(key -> labels[key], data.method), levels=map(key -> labels[key], interest))

p = groupedbar(data.matrix,
    data.speedup,
    group=group,
    xlabel="Dataset",
    ylabel = "Speedup Over OpenCV"
)

savefig(p, "all_pairs.png")