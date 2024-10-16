import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import os
import re

CHARTS_DIRECTORY = "./charts/"  # Ensure this directory exists

def generate_chart_for_operation(path, operation, filename, method_order, matrix_order, baseline_method="spgemm_taco_gustavson", log_scale=False):
    # Load the results from the JSON file
    results = json.load(open(path, 'r'))

    data = defaultdict(lambda: defaultdict(float))
    baseline_times = {}

    # Filter results by the specific operation and prepare data
    for result in results:
        if result["kernel"] != operation:
            continue

        mtx = result["matrix"]
        method = result["method"]
        if method == baseline_method:
            baseline_times[mtx] = result["time"]

    for result in results:
        if result["kernel"] != operation:
            continue

        mtx = result["matrix"]
        method = result["method"]
        if mtx in matrix_order and method in method_order:  # Only include specified matrices and methods
            time = result["time"]
            data[method][mtx] = time
            #speedup = baseline_times[mtx] / time if mtx in baseline_times and time else 0
            #data[method][mtx] = speedup

    filtered_method_order = [method for method in method_order if method in data]
    filtered_matrix_order = [mtx for mtx in matrix_order if any(mtx in data[method] for method in method_order)]
    
    ordered_data = {
        method: [data[method][mtx] for mtx in filtered_matrix_order if mtx in data[method]]
        for method in filtered_method_order
    }
    print(ordered_data)
    print(filtered_matrix_order)

    filtered_matrix_order = [mtx.rsplit('/',1)[-1] for mtx in filtered_matrix_order]
    #Strip everything except the digits from the matrix name
    filtered_matrix_order = [re.sub(r'\D', '', mtx) for mtx in filtered_matrix_order]

    make_line_plot(filtered_method_order, filtered_matrix_order, ordered_data, filename, title=f"SpGEMM Runtime Versus Increasing Dimension (Density = 0.001)", log_scale=log_scale)

def make_line_plot(labels, x_axis, data, filename, title="", y_label="Runtime (s)", log_scale=False):
    x = np.arange(len(x_axis))  # Positions for each matrix
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjust figure size as needed

    method_nice_names = {
        "spgemm_taco_inner": "TACO Inner",
        "spgemm_finch_inner": "Finch Inner",
        "spgemm_taco_gustavson": "TACO Gustavson",
        "spgemm_finch_gustavson": "Finch Gustavson",
        "spgemm_eigen_gustavson": "Eigen",
        "spgemm_mkl_gustavson": "MKL",
        "spgemm_taco_outer": "TACO Outer",
        "spgemm_finch_outer_dense": "Finch Outer Dense",
        "spgemm_finch_outer": "Finch Outer",
        "spgemm_finch_outer_bytemap": "Finch Outer Bytemap",
    }

    # Plotting each method's data as a line
    for label in labels:
        ax.plot(x, data[label], marker='o', label=method_nice_names[label])  # Line plot with markers

    if log_scale:
        ax.set_yscale('log')

    ax.set_ylabel(y_label)
    ax.set_xlabel("Matrix Dimension Size")  # You can modify this based on your x-axis data
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis, rotation=25, ha="right")
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig(CHARTS_DIRECTORY + filename, dpi=200)

# Ensure the charts directory exists or create it
if not os.path.exists(CHARTS_DIRECTORY):
    os.makedirs(CHARTS_DIRECTORY)

matrix_order = [
    "rand_128.mtx",
    "rand_256.mtx",
    "rand_512.mtx",
    "rand_1024.mtx",
    "rand_2048.mtx",
    "rand_4096.mtx",
    "rand_8192.mtx",
    "rand_16384.mtx",
]

method_order = [
    "spgemm_taco_inner",
    "spgemm_finch_inner",
    "spgemm_taco_gustavson",
    "spgemm_finch_gustavson",
    "spgemm_eigen_gustavson",
    "spgemm_mkl_gustavson",
    "spgemm_taco_outer",
    "spgemm_finch_outer_dense",
    "spgemm_finch_outer",
    "spgemm_finch_outer_bytemap",
]

# Example usage, specifying method and matrix order when calling the function
generate_chart_for_operation("scale_results.json", "spgemm", "scaling_spgemm.png", 
                             method_order, matrix_order,
                             baseline_method="spgemm_taco_gustavson", log_scale=True)
