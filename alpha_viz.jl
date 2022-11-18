using JSON
using Statistics

d = open("alpha_results.json", "r") do f
    JSON.parse(f)
end

times = Dict()

for dd in d
    push!(get!(get!(times, dd["dataset"], Dict()), dd["kind"], []), dd["time"])
end

header = [keys(first(times)[2])...]
println("dataset, ", join(header, ", "))
for (dataset, tt) in times
    print(dataset, ", ")
    println(join(mean.(map(kind -> tt[kind], header)), ", "))
end
