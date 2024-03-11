import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict

RESULTS_FILE_PATH = "lanka_data.json"
CHARTS_DIRECTORY = "./charts/"  # Ensure this directory exists

def create_charts_by_operation():
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    operations = defaultdict(list)

    # Group results by operation
    for result in results:
        operations[result["operation"]].append(result)

    # Generate a chart for each operation
    for operation, results in operations.items():
        generate_chart_for_operation(operation, results)

def generate_chart_for_operation(operation, results):
    mtxs = []
    data = defaultdict(list)

    for result in results:
        mtx = result["matrix"]
        method = result["method"]
        if mtx not in mtxs:
            mtxs.append(mtx)

        method = method.replace(".jl", "")  # Simplify method names if needed
        data[method].append(result["time"])

    methods = list(data.keys())
    make_grouped_bar_chart(methods, mtxs, data, title=f"{operation} Performance Comparison")

def make_grouped_bar_chart(labels, x_axis, data, title="", y_label="Time (seconds)"):
    x = np.arange(len(x_axis))
    width = 0.15  # Adjust width based on the number of labels
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust figure size as needed

    for i, label in enumerate(labels):
        offset = width * i
        ax.bar(x + offset, data[label], width, label=label)

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(x_axis, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    fig_file = f"{title.lower().replace(' ', '_').replace('-', '_')}.png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200)
    plt.show()

# Ensure the charts directory exists or create it
import os
if not os.path.exists(CHARTS_DIRECTORY):
    os.makedirs(CHARTS_DIRECTORY)

create_charts_by_operation()
