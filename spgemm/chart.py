import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import os

CHARTS_DIRECTORY = "./charts/"  # Ensure this directory exists

def generate_chart_for_operation(path, operation, filename, baseline_method="spgemm_taco_gustavson", log_scale=False):
    # Load the results from the JSON file
    results = json.load(open(path, 'r'))

    mtxs = []
    data = defaultdict(list)
    baseline_times = {}

    # Filter results by the specific operation and prepare data
    for result in results:
        if result["kernel"] != operation:
            continue

        mtx = result["matrix"]
        method = result["method"]
        if mtx not in mtxs:
            mtxs.append(mtx)
        if method == baseline_method:
            baseline_times[mtx] = result["time"]

    # Calculate speedup relative to baseline
    for result in results:
        if result["kernel"] != operation:
            continue

        mtx = result["matrix"]
        method = result["method"].replace(".jl", "")
        if mtx in baseline_times:
            time = result["time"]
            speedup = baseline_times[mtx] / time if time else 0
            data[method].append(speedup)

    methods = list(data.keys())
    make_grouped_bar_chart(methods, mtxs, data, filename, title=f"{path} Speedup over {baseline_method}", log_scale=log_scale)

def make_grouped_bar_chart(labels, x_axis, data, filename, title="", y_label="Speedup", log_scale=False):
    x = np.arange(len(x_axis))
    width = 0.15  # Adjust width based on the number of labels
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

    for i, label in enumerate(labels):
        offset = width * i
        ax.bar(x + offset, data[label], width, label=label)

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(x_axis, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    plt.savefig(CHARTS_DIRECTORY + filename, dpi=200)

# Ensure the charts directory exists or create it
if not os.path.exists(CHARTS_DIRECTORY):
    os.makedirs(CHARTS_DIRECTORY)

# Example usage, specify the filename when calling the function
generate_chart_for_operation("lanka_joel.json", "spgemm", "lanka_joel_speedup.png", baseline_method="spgemm_taco_gustavson")
generate_chart_for_operation("lanka_small.json", "spgemm", "lanka_small_speedup_log_scale.png", baseline_method="spgemm_taco_gustavson", log_scale=True)
