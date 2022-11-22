using JSON
using DataFrames
#import UnicodePlots
using StatsPlots
unicodeplots()

all_pairs_data = DataFrame(open("all_pairs_results.json", "r") do f
    JSON.parse(f)
end)

all_pairs_data = stack(all_pairs_data, [:opencv_time, :finch_time, :finch_gallop_time, :finch_vbl_time, :finch_rle_time])

all_pairs_labels = Dict(
    "mnist_train" => "MNIST",
    "emnist_letters_train" => "EMNIST Letters",
    "emnist_digits_train" => "EMNIST Digits",
    "omniglot_train" => "Omniglot",
)
all_pairs_data[:, :matrix] = map(key->all_pairs_labels[key], all_pairs_data.matrix)

p = groupedbar(all_pairs_data.matrix,
    all_pairs_data.value,
    group=all_pairs_data.variable,
    xlabel="Dataset",
    ylabel = "Speedup Over OpenCV",
    #xticks = (1:length(unique(all_pairs_data.matrix)),
    #all_pairs_data.matrix),
)

display(p)
#=
savefig(p, "all_pairs.png")
=#

#=
@df all_pairs_data begin
    groupedbar!(p, :finch_time ./ :opencv_time, bar_position=:dodge)
    groupedbar!(p, :finch_gallop_time ./ :opencv_time)
end



    k
    "Finch" => all_pairs_data.finch_time ./ all_pairs_data.opencv_time,
    "Finch (Gallop)" => all_pairs_data.finch_gallop_time ./ all_pairs_data.opencv_time,
    "Finch (VBL)" => all_pairs_data.finch_vbl_time ./ all_pairs_data.opencv_time,
    "Finch (RLE)" => all_pairs_data.finch_rle_time ./ all_pairs_data.opencv_time
)

p = bar(

display(bar(
    1:nrow(all_pairs_plot_data),
    Matrix(all_pairs_plot_data),
    xticks = (1:nrow(all_pairs_plot_data),
    all_pairs_data.matrix),
    x="Dataset",
    ylabel = "Speedup Over OpenCV"))
=#


#=
frame = pd.read_csv('ReadyAllPairs.csv')
viz = frame.plot(kind="bar", figsize=(10,3.5), x="Dataset", ylabel = "Speedup Over OpenCV", rot=0, yticks = [y*0.2 for y in range(6)], color=color)
viz.axes.get_xaxis().get_label().set_visible(False)
viz.axes.axhline(1, color="grey")
viz.legend(loc="upper left")
viz.get_figure().savefig("images/allpairs.png", bbox_inches="tight")







headers = ["matrix", "opencv_time", "finch_time", "finch_gallop_time", "finch_vbl_time", "finch_rle_time"]

println(join(headers, ", "))
for dd in d
    println(join(map(key -> dd[key], headers), ", "))
end

=#
