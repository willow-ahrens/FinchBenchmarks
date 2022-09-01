using JSON


d = open("all_pairs_results.json", "r") do f
    JSON.parse(f)
end

headers = ["matrix", "opencv_time", "finch_time", "finch_gallop_time", "finch_vbl_time", "finch_rle_time"]

println(join(headers, ", "))
for dd in d
    println(join(map(key -> dd[key], headers), ", "))
end

