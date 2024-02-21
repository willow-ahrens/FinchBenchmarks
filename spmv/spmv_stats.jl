using JSON

function calc_speedup(slow_method, fast_method)
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
        count += 1
        speedup += times[slow_method] / times[fast_method]
    end

    return speedup / count
end