using JSON


d = open("conv_results.json", "r") do f
    JSON.parse(f)
end

headers = ["p", "opencv_time", "finch_time"]

println(join(headers, ", "))
for dd in d
    println(join(map(key -> dd[key], headers), ", "))
end

