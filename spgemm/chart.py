import matplotlib.pyplot as plt
import numpy as np
import json
from collections import defaultdict
import os
import re

CHARTS_DIRECTORY = "./charts/"  # Ensure this directory exists

def generate_chart_for_operation(path, operation, filename, method_order, matrix_order, baseline_method="spgemm_taco_gustavson", log_scale=False, title=""):
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
            speedup = baseline_times[mtx] / time if mtx in baseline_times and time else 0
            data[method][mtx] = speedup

    filtered_method_order = [method for method in method_order if method in data]
    filtered_matrix_order = [mtx for mtx in matrix_order if any(mtx in data[method] for method in method_order)]
    
    ordered_data = {
        method: [data[method][mtx] for mtx in filtered_matrix_order if mtx in data[method]]
        for method in filtered_method_order
    }
    print(ordered_data)
    print(filtered_matrix_order)

    filtered_matrix_order = [mtx.rsplit('/',1)[-1] for mtx in filtered_matrix_order]

    make_grouped_bar_chart(filtered_method_order, filtered_matrix_order, ordered_data, filename, title=title, log_scale=log_scale)

def make_grouped_bar_chart(labels, x_axis, data, filename, title="", y_label="Speedup", log_scale=False):
    x = np.arange(len(x_axis))
    width = 0.7/len(labels)  # Adjust width based on the number of labels
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

    for i, label in enumerate(labels):
        offset = width * (i - len(labels)/2)  # Center bars around the tick
        #cross-hatch the bar if the string "finch" is not in the label
        ax.bar(x + offset, data[label], width, label=method_nice_names[label])

    ax.axhline(y=1, color='r', linestyle='--', linewidth=1)

    if log_scale:
        ax.set_yscale('log')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis, rotation=25, ha="right")
    ax.legend(ncol=4)

    plt.tight_layout()
    plt.savefig(CHARTS_DIRECTORY + filename, dpi=200)

# Ensure the charts directory exists or create it
if not os.path.exists(CHARTS_DIRECTORY):
    os.makedirs(CHARTS_DIRECTORY)

matrix_order = [
"SNAP/email-Eu-core", 
"SNAP/CollegeMsg", 
"SNAP/soc-sign-bitcoin-alpha", 
"SNAP/ca-GrQc", 
"SNAP/soc-sign-bitcoin-otc", 
"SNAP/p2p-Gnutella08", 
"SNAP/as-735", 
"SNAP/p2p-Gnutella09", 
"SNAP/wiki-Vote", 
"SNAP/p2p-Gnutella06", 
"SNAP/p2p-Gnutella05", 
"SNAP/ca-HepTh", 
"FEMLAB/poisson3Da", 
"SNAP/ca-CondMat", 
"SNAP/email-Enron", 
"SNAP/p2p-Gnutella31", 
"Um/2cubes_sphere", 
"Oberwolfach/filter3D", 
"Williams/cop20k_A", 
"vanHeukelum/cage12", 
"Hamm/scircuit", 
"JGD_Homology/m133-b3", 
"Pajek/patents_main", 
"Um/offshore", 
"GHS_indef/mario002", 
"SNAP/amazon0312", 
"SNAP/web-Google", 
"Williams/webbase-1M", 
"SNAP/roadNet-CA", 
"SNAP/cit-Patents"
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
generate_chart_for_operation("lanka_joel.json", "spgemm", "spgemm_joel_speedup_log_scale.png", 
                             method_order, matrix_order,
                             baseline_method="spgemm_taco_gustavson", log_scale=True, title="SpGEMM Speedup Over Taco Gustavson on Large Matrices")

generate_chart_for_operation("lanka_small.json", "spgemm", "spgemm_small_speedup_log_scale.png", 
                             method_order, matrix_order,
                             baseline_method="spgemm_taco_gustavson", log_scale=True, title="SpGEMM Speedup Over Taco Gustavson on Small Matrices")