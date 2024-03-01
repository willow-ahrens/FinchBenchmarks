import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict

def create_chart():
    results = json.load(open('spmv_results.json', 'r'))
    mtxs = []
    data = defaultdict(list)
    finch_formats = get_best_finch_format()

    for result in results:
        mtx = result["matrix"]
        method = result["method"]
        if mtx not in mtxs:
            mtxs.append(mtx)

        if "finch" in method and finch_formats[mtx] != method:
            continue
        method = "finch" if "finch" in method else method
        data[method].append(result["time"])

    x = np.arange(len(mtxs))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()

    methods = ["julia", "finch", "taco", "suite_sparse"]

    ref  = "julia"
    ref_data = data[ref]
    for method in methods:
        method_data = data[method]
        data[method] = [ref_data[i] / method_data[i] for i in range(len(ref_data))] 

    for method, times in data.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, times, width, label=method)
        labels = [round(float(time), 2) if time != 1 and method == "finch" else "" for time in times]
        ax.bar_label(rects, padding=3, labels=labels, fontsize=6)
        multiplier += 1

    ax.set_ylabel('Speedup')
    ax.set_title('SpMV Performance')
    ax.set_xticks(x + width*1.5, mtxs)
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=45)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, 2.5)

    plt.show()

def get_best_finch_format():
    results = json.load(open('spmv_results.json', 'r'))
    formats = defaultdict(list)
    for result in results:
        if "finch" not in result["method"]:
            continue
        formats[result["matrix"]].append((result["method"], result["time"]))

    best_formats = defaultdict(list)
    for matrix, format_times in formats.items():
        best_format, _ = min(format_times, key=lambda x: x[1])
        best_formats[matrix] = best_format
    
    return best_formats



create_chart()
# print(get_best_finch_format())


