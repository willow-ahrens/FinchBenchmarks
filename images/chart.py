import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import os
from scipy.stats import gmean

RESULTS_FILE_PATH = "lanka_data.json"
CHARTS_DIRECTORY = "./charts/"  # Ensure this directory exists


def generate_chart_for_operation(operation, baseline_method="opencv", log_scale=False, excluded_datasets=None, title=""):
    if excluded_datasets is None:
        excluded_datasets = []
    # Load the results from the JSON file
    results = json.load(open(RESULTS_FILE_PATH, 'r'))

    datasets = set()
    data = defaultdict(lambda: defaultdict(list))
    baseline_times = {}


    # Filter results by the specific operation and prepare data
    for result in results:
        if result["operation"] != operation or result["dataset"] in excluded_datasets:
            continue

        dataset = result["dataset"]
        label = result["label"]
        method = result["method"]
        datasets.add(dataset)
        if method == baseline_method:
            baseline_times[(dataset, label)] = result["time"]

    # Calculate speedup relative to baseline
    for result in results:
        if result["operation"] != operation or result["dataset"] in excluded_datasets:
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

    make_grouped_bar_chart(methods, datasets, geomean_data, title=title)

def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        #ax.annotate(
        #    label,                      # Use `label` as label
        #    (x_value, y_value),         # Place label at end of the bar
        #    xytext=(0, space),          # Vertically shift label by `space`
        #    textcoords="offset points", # Interpret `xytext` as offset in points
        #    ha='center',                # Horizontally center label
        #    va=va)                      # Vertically align label differently for
        #                                # positive and negative values.

def make_grouped_bar_chart(labels, x_axis, data, title="", y_label="Speedup", log_scale=False):
    x = np.arange(len(x_axis))  # the label locations
    num_labels = len(labels)
    width = 0.8 / num_labels  # the width of the bars, adjust to fit

    method_labels = {
        "opencv": "OpenCV",
        "finch": "Finch (Naive)",
        "finch_rle": "Finch (RunList)",
        "finch_bits": "Finch (Bitwise)",
        "finch_bits_mask": "Finch (Bitwise + Mask)"
    }


    fig, ax = plt.subplots(figsize=(12, 6))
    for i, label in enumerate(labels):
        speeds = [data[label].get(dataset, 0) for dataset in x_axis]
        ax.bar(x + i*width - width*(num_labels-1)/2, speeds, width, label=method_labels.get(label, label))

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    #ax.set_xticklabels(x_axis, rotation=45, ha="right")
    new_labels = {"mnist": "MNIST", "omniglot": "Omniglot", "humansketches":
    "Sketches", "testimage_dip3e": "dip3e", "mnist_magnify": "MNIST 8X",
    "omniglot_magnify": "Omniglot 8X", "humansketches_magnify": "Sketches 8X",
    "testimage_dip3e_magnify": "dip3e 8X"}
    x_axis = [new_labels[dataset] for dataset in x_axis]
    ax.set_xticklabels(x_axis, font = {'size': 12})
    ax.legend()

    if log_scale:
        ax.set_yscale('log')  # Set the y-axis to a logarithmic scale

    # Add a horizontal dashed red line at y=1.0
    ax.axhline(y=1.0, color='red', linestyle='--')

    plt.tight_layout()
    fig_file = f"{title.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}.png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200)
    add_value_labels(ax)
    plt.show()



# Ensure the charts directory exists or create it
if not os.path.exists(CHARTS_DIRECTORY):
    os.makedirs(CHARTS_DIRECTORY)

# Generate charts for each operation by calling the function with the operation and baseline method
generate_chart_for_operation("erode2", baseline_method="opencv", log_scale=True, excluded_datasets=["mnist", "omniglot"], title = "Speedup over OpenCV on 2 Iterations of Erosion")
generate_chart_for_operation("erode4", baseline_method="opencv", log_scale=True, excluded_datasets=["mnist", "omniglot"], title="Speedup over OpenCV on 4 Iterations of Erosion")
generate_chart_for_operation("hist", baseline_method="opencv", log_scale=True, title="Speedup over OpenCV on Masked Histogram")
