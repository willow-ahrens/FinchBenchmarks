import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import os
from scipy.stats import gmean

RESULTS_FILE_PATH = "lanka_data.json"
CHARTS_DIRECTORY = "./charts/"  # Ensure this directory exists

def generate_chart_for_operation(operation, baseline_method="opencv", log_scale=False):
    # Load the results from the JSON file
    results = json.load(open(RESULTS_FILE_PATH, 'r'))

    datasets = set()
    data = defaultdict(lambda: defaultdict(list))
    baseline_times = {}

    # Filter results by the specific operation and prepare data
    for result in results:
        if result["operation"] != operation:
            continue

        dataset = result["dataset"]
        label = result["label"]
        method = result["method"]
        datasets.add(dataset)
        if method == baseline_method:
            baseline_times[(dataset, label)] = result["time"]

    # Calculate speedup relative to baseline
    for result in results:
        if result["operation"] != operation:
            continue

        dataset = result["dataset"]
        label = result["label"]
        method = result["method"].replace(".jl", "")
        if method != baseline_method and (dataset, label) in baseline_times:
            time = result["time"]
            speedup = baseline_times[(dataset, label)] / time if time else 0
            data[method][dataset].append(speedup)

    # Calculate geometric mean for each method across all datasets
    geomean_data = {}
    for method, datasets in data.items():
        geomean_data[method] = {dataset: gmean(speedups) for dataset, speedups in datasets.items()}

    # Plot
    #datasets = sorted(datasets)  # Sort datasets for consistent plotting
    methods = sorted(geomean_data.keys())
    make_grouped_bar_chart(methods, datasets, geomean_data, title=f"{operation} Speedup over {baseline_method}", log_scale=log_scale)

def make_grouped_bar_chart(labels, x_axis, data, title="", y_label="Speedup", log_scale=False):
    x = np.arange(len(x_axis))  # the label locations
    num_labels = len(labels)
    width = 0.8 / num_labels  # the width of the bars, adjust to fit

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, label in enumerate(labels):
        speeds = [data[label].get(dataset, 0) for dataset in x_axis]
        ax.bar(x + i*width - width*(num_labels-1)/2, speeds, width, label=label)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis, rotation=45, ha="right")
    ax.legend()

    if log_scale:
        ax.set_yscale('log')  # Set the y-axis to a logarithmic scale

    plt.tight_layout()
    fig_file = f"{title.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}.png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200)

# Ensure the charts directory exists or create it
if not os.path.exists(CHARTS_DIRECTORY):
    os.makedirs(CHARTS_DIRECTORY)

# Generate charts for each operation by calling the function with the operation and baseline method
generate_chart_for_operation("erode2", baseline_method="opencv", log_scale=True)
generate_chart_for_operation("erode4", baseline_method="opencv", log_scale=True)
generate_chart_for_operation("hist", baseline_method="opencv", log_scale=True)
