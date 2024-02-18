using JSON

spmv_results_text = read("spmv_results.json", String)
spmv_results = JSON.parse(spmv_results_text)

kernel_times = Dict()
for result in spmv_results
    kernel = result["matrix"]
    method = result["method"]
    time = result["time"]

    times = get(kernel_times, kernel, Dict())
    times[method] = time
    kernel_times[kernel] = times
end

count = 0
speedup = 0
for (kernel, times) in kernel_times
    global count += 1
    global speedup += times["julia"] / times["finch"]
end

println(speedup / count)