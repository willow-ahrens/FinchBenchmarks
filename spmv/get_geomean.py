import json
import math
import argparse
from collections import defaultdict

def geometric_mean(arr):
    n = len(arr)
    return math.exp(sum(math.log(x) for x in arr) / n)

def main():
    filename = 'spmv_results.json'
    with open(filename, 'r') as file:
        data = json.load(file)

    times_by_matrix = defaultdict(dict)
    for entry in data:
        matrix = entry['matrix']
        method = entry['method']
        time = entry['time']
        times_by_matrix[matrix][method] = time

    baseline_method = 'finch_unsym'
    speedup_by_method = defaultdict(list)

    for matrix, times in times_by_matrix.items():
        baseline_time = times.get(baseline_method)
        if baseline_time is not None:
            for method, time in times.items():
                speedup = baseline_time / time
                speedup_by_method[method].append(speedup)

    geo_speedups = {}
    for method, speedups in speedup_by_method.items():
        geo_speedups[method] = 1 / geometric_mean(speedups)

    print("Geometric Speedups:")
    for method, geo_speedup in geo_speedups.items():
        print(f"{method}: {geo_speedup:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate geometric speedup from JSON results.")
    parser.add_argument("filename", type=str, help="The path to the JSON file containing the results.")

    main()
