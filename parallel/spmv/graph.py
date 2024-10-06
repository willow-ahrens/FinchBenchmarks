import json
import math
from collections import defaultdict

import matplotlib.pyplot as plt

GRAPH_FOLDER = "graph"
SPEEDUP_FOLDER = "speedup"
PERFORMANCE_FOLDER = "performance"
RESULTS_FOLDER = "results"

NTHREADS = [1, 2, 3, 4]

METHODS = ["reference", "parallel_row"]

DATASETS = [
    {"uniform": ["1000-0.1", "10000-0.1"]},
    {"FEMLAB": ["FEMLAB-poisson3Da", "FEMLAB-poisson3Db"]},
]

COLORS = ["gray", "cadetblue", "saddlebrown", "navy"]


def load_json():
    combine_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))
    for n_thread in NTHREADS:
        results_json = json.load(
            open(f"{RESULTS_FOLDER}/spmv_{n_thread}_threads.json", "r")
        )
        for result in results_json:

            matrix = (
                result["matrix"].replace("/", "-")
                if result["dataset"] != "uniform"
                else f"{result['matrix']['size']}-{result['matrix']['sparsity']}"
            )
            combine_results[result["dataset"]][matrix][result["method"]][
                result["n_threads"]
            ] = result["time"]

    return combine_results


def plot_speedup_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 6))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [
                results[dataset][matrix][method][1]
                / results[dataset][matrix][method][n_thread]
                for n_thread in NTHREADS
            ],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"Speedup for {dataset}: {matrix}")
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Speedup")

    plt.legend()
    plt.savefig(save_location)


def plot_performance_result(results, dataset, matrix, save_location):
    plt.figure(figsize=(10, 6))
    for method, color in zip(METHODS, COLORS):
        plt.plot(
            NTHREADS,
            [
                math.log(results[dataset][matrix][method][n_thread], 10)
                for n_thread in NTHREADS
            ],
            label=method,
            color=color,
            marker="o",
            linestyle="-",
            linewidth=1,
        )

    plt.title(f"Performance for {dataset}: {matrix}")
    plt.xticks(NTHREADS)
    plt.xlabel("Number of Threads")
    plt.ylabel(f"Log of Runtime (second)")

    plt.legend()
    plt.savefig(save_location)


if __name__ == "__main__":
    results = load_json()
    for datasets in DATASETS:
        for dataset, matrices in datasets.items():
            for matrix in matrices:
                plot_speedup_result(
                    results,
                    dataset,
                    matrix,
                    f"{GRAPH_FOLDER}/{SPEEDUP_FOLDER}/{dataset}-{matrix}.png",
                )
                plot_performance_result(
                    results,
                    dataset,
                    matrix,
                    f"{GRAPH_FOLDER}/{PERFORMANCE_FOLDER}/{dataset}-{matrix}.png",
                )
