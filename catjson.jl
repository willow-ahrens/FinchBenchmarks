using JSON

res = []
for arg in ARGS
    append!(res, JSON.parsefile(arg))
end
println(json(res, 4))
