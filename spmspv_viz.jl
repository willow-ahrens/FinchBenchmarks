using JSON


d = open("spmspv_hb.json", "r") do f
    JSON.parse(f)
end

headers = ["matrix", "taco_times", "finch_times", "finch_gallop_times", "finch_lead_times", "finch_follow_times", "finch_vbl_times"]

println(join(headers, ", "))
for dd in d
    if dd["x"] == "0.1 density"
        println(join(map(key -> key == "matrix" ? dd[key] : first(dd[key]), headers), ", "))
    end
end

