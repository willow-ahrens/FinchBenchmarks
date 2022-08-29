using JSON


d = open("triangle_results.json", "r") do f
    JSON.parse(f)
end

headers = ["matrix", "taco_time", "finch_time", "finch_gallop_time"]

println(join(headers, ", "))
for dd in d
    println(join(map(key -> dd[key], headers), ", "))
end

