import json
import math
import argparse
from collections import defaultdict

def geometric_mean(arr):
    n = len(arr)
    return math.exp(sum(math.log(x) for x in arr) / n)

def main(filename):
    with open(filename, 'r') as file:
        data = json.load(file)

    times_by_matrix = defaultdict(dict)
    for entry in data:
        matrix = entry['matrix']
        method = entry['method']
        time = entry['time']
        times_by_matrix[matrix][method] = time


    for matrix, times in times_by_matrix.items():
        best_finch_method = None
        best_finch_time = float('inf')
        for method, time in times.items():
            if method.startswith('finch') and time < best_finch_time:
                best_finch_time = time
                best_finch_method = method
        if best_finch_method is not None:
            times_by_matrix[matrix]['best_finch_method'] = best_finch_time

    for matrix, times in times_by_matrix.items():
        best_finch_method = None
        best_finch_time = float('inf')
        for method, time in times.items():
            if method.startswith('finch_unsym') and time < best_finch_time:
                best_finch_time = time
                best_finch_method = method
        if best_finch_method is not None:
            times_by_matrix[matrix]['baseline_finch'] = best_finch_time

    for matrix, times in times_by_matrix.items():
        best_finch_method = None
        best_finch_time = float('inf')
        for method, time in times.items():
            if method.startswith('taco') and time < best_finch_time:
                best_finch_time = time
                best_finch_method = method
        if best_finch_method is not None:
            times_by_matrix[matrix]['best_taco_method'] = best_finch_time

    baseline_method = 'best_taco_method'
    speedup_by_method = defaultdict(list)

    for matrix, times in times_by_matrix.items():
        baseline_time = times.get(baseline_method)
        if baseline_time is not None:
            for method, time in times.items():
                speedup = baseline_time/time
                print(f"{matrix}: {method}: {speedup}")
                speedup_by_method[method].append(speedup)

    geo_speedups = {}
    for method, speedups in speedup_by_method.items():
        geo_speedups[method] = geometric_mean(speedups)

    print("Geometric Speedups:")
    for method, geo_speedup in geo_speedups.items():
        print(f"{method}: {geo_speedup:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate geometric speedup from JSON results.")
    parser.add_argument("filename", type=str, help="The path to the JSON file containing the results.")

    args = parser.parse_args()

    main(args.filename)
