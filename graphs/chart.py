import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import numpy as np
import json
from collections import defaultdict
import os

RESULTS_FILE_PATH = "lanka_data.json"
CHARTS_DIRECTORY = "./charts/"  # Ensure this directory exists

def generate_chart_for_operation(operation, baseline_method="Graphs.jl", log_scale=False):
    # Load the results from the JSON file
    results = json.load(open(RESULTS_FILE_PATH, 'r'))

    mtxs = []
    data = defaultdict(list)
    baseline_times = {}

    # Filter results by the specific operation and prepare data
    for result in results:
        if result["operation"] != operation:
            continue

        mtx = result["matrix"]
        method = result["method"]
        if mtx not in mtxs:
            mtxs.append(mtx)
        if method == baseline_method:
            baseline_times[mtx] = result["time"]

    # Calculate speedup relative to baseline
    for result in results:
        if result["operation"] != operation:
            continue

        mtx = result["matrix"]
        method = result["method"].replace(".jl", "")
        if method != baseline_method and mtx in baseline_times:
            time = result["time"]
            speedup = baseline_times[mtx] / time if time else 0
            data[method].append(speedup)

    mtxs = [mtx.rsplit('/',1)[-1] for mtx in mtxs]
    methods = list(data.keys())
    make_grouped_bar_chart(methods, mtxs, data, title=f"{operation} Speedup over {baseline_method}", log_scale=log_scale)

def make_grouped_bar_chart(labels, x_axis, data, title="", y_label="Speedup", log_scale=False):
    x = np.arange(len(x_axis))
    width = 0.15  # Adjust width based on the number of labels
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size as needed

    for i, label in enumerate(labels):
        offset = width * i
        ax.bar(x + offset, data[label], width, label=label)

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    #ax.set_xticklabels(x_axis, rotation=45, ha="right")
    ax.set_xticklabels(x_axis, font = {'size': 12})
    ax.legend()

    plt.tight_layout()
    fig_file = f"{title.lower().replace(' ', '_').replace('-', '_')}.png"

    # Add a horizontal dashed red line at y=1.0
    ax.axhline(y=1.0, color='red', linestyle='--')

    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200)
    plt.show()

# Ensure the charts directory exists or create it
if not os.path.exists(CHARTS_DIRECTORY):
    os.makedirs(CHARTS_DIRECTORY)

# Generate charts for each operation by calling the function with the operation, baseline method, and log scale parameter
generate_chart_for_operation("bfs", baseline_method="Graphs.jl")
generate_chart_for_operation("bellmanford", baseline_method="Graphs.jl")
