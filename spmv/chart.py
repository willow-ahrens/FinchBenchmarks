import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import math
from collections import defaultdict
import re

RESULTS_FILE_PATH = "spmv_results_lanka.json"
CHARTS_DIRECTORY = "charts/"
FORMAT_ORDER = {
    "finch": -1,
    "finch_unsym": -2,
    "finch_unsym_row_maj": -3,
    "finch_vbl": -4,
    "finch_vbl_unsym": -5,
    "finch_vbl_unsym_row_maj": -6,
    "finch_band": -7,
    "finch_band_unsym": -8,
    "finch_band_unsym_row_maj": -9,
    "finch_pattern": -10,
    "finch_pattern_unsym": -11,
    "finch_pattern_unsym_row_maj": -12,
    "finch_point": -13,
    "finch_point_row_maj": -14,
    "finch_point_pattern": -15,
    "finch_point_pattern_row_maj": -16,
    "finch_blocked": -17,
}
FORMAT_LABELS = {
    "finch": "Symmetric SparseList",
    "finch_unsym": "SparseList",
    "finch_unsym_row_maj": "SparseList (Row-Major)",
    "finch_vbl": "Symmetric SparseVBL",
    "finch_vbl_unsym": "SparseVBL",
    "finch_vbl_unsym_row_maj": "SparseVBL (Row-Major)",
    "finch_band": "Symmetric SparseBand",
    "finch_band_unsym": "SparseBand",
    "finch_band_unsym_row_maj": "SparseBand (Row-Major)",
    "finch_pattern": "Symmetric Pattern",
    "finch_pattern_unsym": "Pattern",
    "finch_pattern_unsym_row_maj": "Pattern (Row-Major)",
    "finch_point": "SparsePoint",
    "finch_point_row_maj": "SparsePoint (Row-Major)",
    "finch_point_pattern": "SparsePoint Pattern",
    "finch_point_pattern_row_maj": "SparsePoint Pattern (Row-Major)",
    "finch_blocked": "4D-Blocked"
}

def all_formats_chart(ordered_by_format=False):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    data = defaultdict(lambda: defaultdict(int))
    finch_formats = get_best_finch_format()

    for result in results:
        mtx = result["matrix"]
        method = result["method"]

        if "finch" in method and finch_formats[mtx] != method:
            continue
        method = "finch" if "finch" in method else method
        data[mtx][method] = result["time"]

    for mtx, times in data.items():
        ref_time = times["taco"]
        for method, time in times.items():
            times[method] = ref_time / time

    if ordered_by_format:
        #ordered_data = sorted(data.items(), key = lambda mtx_results: (mtx_results[1]["finch"] > 1, FORMAT_ORDER[finch_formats[mtx_results[0]]], mtx_results[1]["finch"]), reverse=True)
        ordered_data = sorted(data.items(), key = lambda mtx_results: (FORMAT_ORDER[finch_formats[mtx_results[0]]], mtx_results[1]["finch"]), reverse=True)
    else:
        ordered_data = sorted(data.items(), key = lambda mtx_results: mtx_results[1]["finch"], reverse=True)

    #faster_data = defaultdict(list)
    #slower_data = defaultdict(list)
    all_data = defaultdict(list)
    #splice_idx = 0
    for i, (mtx, times) in enumerate(ordered_data):
        for method, time in times.items():
            all_data[method].append(time)
            #if times["finch"] > 1.0:
            #    splice_idx = max(splice_idx, i + 1)
            #    faster_data[method].append(time)
            #else:
            #    slower_data[method].append(time)

    ordered_mtxs = [mtx for mtx, _ in ordered_data]
    labels = [FORMAT_LABELS[finch_formats[mtx]] for mtx, _ in ordered_data]
    #methods = ["finch", "julia_stdlib", "suite_sparse"]
    methods = ["finch", "julia_stdlib"]
    colors = {"finch": "tab:blue", "julia_stdlib": "tab:orange", "suite_sparse": "tab:green"}

    short_mtxs = [mtx.rsplit('/',1)[-1] for mtx in ordered_mtxs]
    new_mtxs = {
        "toeplitz_large_band": "large_band",
        "toeplitz_medium_band": "medium_band",
        "toeplitz_small_band": "small_band",
        #"TSOPF_RS_b678_c1": "*RS_b678_c1",
    }
    short_mtxs = [new_mtxs.get(mtx, mtx) for mtx in short_mtxs]

    make_grouped_bar_chart(methods, short_mtxs, all_data, colors=colors, labeled_groups=["finch"], bar_labels_dict={"finch": labels[:]}, title="SpMV Performance (Speedup Over Taco) labeled")
    make_grouped_bar_chart(methods, short_mtxs, all_data, colors=colors, title="SpMV Performance (Speedup Over Taco)")

    #if ordered_by_format:
    #    make_grouped_bar_chart(methods, ordered_mtxs[:splice_idx], faster_data, colors=colors, labeled_groups=["finch"], bar_labels_dict={"finch": labels[:splice_idx]}, title="SpMV Performance (Faster than TACO) Labeled")
    #    make_grouped_bar_chart(methods, ordered_mtxs[splice_idx:], slower_data, colors=colors, labeled_groups=["finch"], bar_labels_dict={"finch": labels[splice_idx:]}, title="SpMV Performance (Slower than TACO) Labeled")
    #    make_grouped_bar_chart(methods, ordered_mtxs[:splice_idx], faster_data, colors=colors, title="SpMV Performance (Faster than TACO)")
    #    make_grouped_bar_chart(methods, ordered_mtxs[splice_idx:], slower_data, colors=colors, title="SpMV Performance (Slower than TACO)")
    #else:
    #    make_grouped_bar_chart(methods, ordered_mtxs[:splice_idx], faster_data, colors=colors, title="SpMV Performance Sorted (Faster than TACO)")
    #    make_grouped_bar_chart(methods, ordered_mtxs[splice_idx:], slower_data, colors=colors, title="SpMV Performance Sorted (Slower than TACO)")
    

    # for mtx in mtxs:
        # all_formats_for_matrix_chart(mtx)


def get_best_finch_format():
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
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


def get_method_results(method, mtxs=[]):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    mtx_times = {}
    for result in results:
        if result["method"] == method and (mtxs == [] or result["matrix"] in mtxs):
            mtx_times[result["matrix"]] = result["time"]
    return mtx_times


def get_speedups(faster_results, slower_results):
    speedups = {}
    for mtx, slow_time in slower_results.items():
        if mtx in faster_results:
            speedups[mtx] = slow_time / faster_results[mtx]
    return speedups


def order_speedups(speedups):
    ordered = [(mtx, time) for mtx, time in speedups.items()]
    return sorted(ordered, key=lambda x: x[1], reverse=True)


def method_to_ref_comparison_chart(method, ref, title=""):
    method_results = get_method_results(method)
    ref_results = get_method_results("taco")
    speedups = get_speedups(method_results, ref_results)

    x_axis = []
    data = defaultdict(list)
    for matrix, speedup in speedups.items():
        x_axis.append(matrix)
        data[method].append(speedup)
        data[ref].append(1)

    make_grouped_bar_chart([method, ref], x_axis, data, labeled_groups=[method], title=title)


def all_formats_for_matrix_chart(matrix):
    results = json.load(open(RESULTS_FILE_PATH, 'r'))
    data = {}
    for result in results:
        if result["matrix"] == matrix:
            data[result["method"]] = result["time"]
    
    formats = []
    speedups = []
    bar_colors = []
    for format, time in data.items():
        formats.append(format)
        speedups.append(data["taco"] / time)
        bar_colors.append("orange" if "finch" in format else "green")
    
    fig, ax = plt.subplots()
    ax.bar(formats, speedups, width = 0.2, color = bar_colors)
    ax.set_ylabel("Speedup")
    ax.set_title(matrix + " Performance")
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)

    fig_file = matrix.lower().replace("/", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + "/matrices/" + fig_file, dpi=200, bbox_inches="tight")
    plt.close() 


def make_grouped_bar_chart(labels, x_axis, data, colors = None, labeled_groups = [], title = "", y_label = "", bar_labels_dict={}):
    x = np.arange(len(data[labels[0]]))
    width = 0.2 
    multiplier = 0
    max_height = 0

    fig, ax = plt.subplots(figsize=(16, 6))
    for label in labels:
        label_data = data[label]
        max_height = max(max_height, max(label_data))
        offset = width * multiplier
        if colors:
            rects = ax.bar(x + offset, label_data, width, label=label, color=colors[label])
        else:
            rects = ax.bar(x + offset, label_data, width, label=label)
        bar_labels = bar_labels_dict[label] if (label in bar_labels_dict) else [round(float(val), 2) if label in labeled_groups else "" for val in label_data]
        ax.bar_label(rects, padding=4, labels=bar_labels, fontsize=5, rotation=90)
        multiplier += 1

    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + width * (len(labels) - 1)/2, x_axis)
    ax.tick_params(axis='x', which='major', labelsize=6, labelrotation=90)
    ax.legend(loc='upper left', ncols=4)
    ax.set_ylim(0, max_height + 0.5)

    plt.plot([-1, len(x_axis)], [1, 1], linestyle='--', color="tab:red", linewidth=0.75)

    fig_file = title.lower().replace(" ", "_") + ".png"
    plt.savefig(CHARTS_DIRECTORY + fig_file, dpi=200, bbox_inches="tight")
    plt.close()
    

all_formats_chart()
all_formats_chart(ordered_by_format=True)
# method_to_ref_comparison_chart("finch", "taco", title="Finch SparseList Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_unsym", "taco", title="Finch SparseList SpMV Performance")
# method_to_ref_comparison_chart("finch_unsym_row_maj", "taco", title="Finch SparseList Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_vbl", "taco", title="Finch SparseVBL Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_vbl_unsym", "taco", title="Finch SparseVBL SpMV Performance")
# method_to_ref_comparison_chart("finch_vbl_unsym_row_maj", "taco", title="Finch SparseVBL Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_band", "taco", title="Finch SparseBand Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_band_unsym", "taco", title="Finch SparseBand SpMV Performance")
# method_to_ref_comparison_chart("finch_band_unsym_row_maj", "taco", title="Finch SparseBand Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_pattern", "taco", title="Finch SparseList Pattern Symmetric SpMV Performance")
# method_to_ref_comparison_chart("finch_pattern_unsym", "taco", title="Finch SparseList Pattern SpMV Performance")
# method_to_ref_comparison_chart("finch_pattern_unsym_row_maj", "taco", title="Finch SparseList Pattern Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_point", "taco", title="Finch SparsePoint SpMV Performance")
# method_to_ref_comparison_chart("finch_point_row_maj", "taco", title="Finch SparsePoint Row Major SpMV Performance")
# method_to_ref_comparison_chart("finch_point_pattern", "taco", title="Finch SparsePoint Pattern SpMV Performance")
# method_to_ref_comparison_chart("finch_point_pattern_row_maj", "taco", title="Finch SparsePoint Pattern Row Major SpMV Performance")