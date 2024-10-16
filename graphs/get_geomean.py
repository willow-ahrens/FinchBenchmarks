import json
import math
import argparse
from collections import defaultdict

def geometric_mean(arr):
    n = len(arr)
    return math.exp(sum(math.log(x) for x in arr) / n)

def calculate_speedups(data, operation):
    times_by_matrix = defaultdict(dict)

    # Group times by matrix for the given operation
    for entry in data:
        if entry['operation'] == operation:
            matrix = entry['matrix']
            method = entry['method']
            time = entry['time']
            times_by_matrix[matrix][method] = time

    # Calculate speedups for each pair of methods
    speedup_by_method_pair = defaultdict(list)

    for matrix, times in times_by_matrix.items():
        methods = list(times.keys())

        # Compare each method with every other method
        for i, baseline_method in enumerate(methods):
            for j, comparison_method in enumerate(methods):
                if i != j:  # Skip comparing a method to itself
                    baseline_time = times[baseline_method]
                    comparison_time = times[comparison_method]
                    if comparison_time > 0:
                        speedup = baseline_time / comparison_time
                        speedup_by_method_pair[(baseline_method, comparison_method)].append(speedup)

    # Compute geometric mean of the speedups for each method pair
    geo_speedups = {}
    for (baseline_method, comparison_method), speedups in speedup_by_method_pair.items():
        geo_speedups[(baseline_method, comparison_method)] = geometric_mean(speedups)

    return geo_speedups

def main(filename):
    # Read data from the JSON file
    with open(filename, 'r') as file:
        datas = json.load(file)

    mtx_order = ["soc-orkut", "soc-LiveJournal1", "hollywood-2009", "indochina-2004", "kron_g500-logn16", "rmat_s22_e64", "rmat_s23_e32", "rmat_s24_e16", "rgg_n_2_24_s0", "roadNet-CA", "road_usa"]

    # Apply rsplit to the mtx fields of datas
    for data in datas:
        data["matrix"] = data["matrix"].rsplit('/', 1)[-1]

    # Sort and filter datas by the order in mtx_order
    datas = [data for data in datas if data["matrix"] in mtx_order]
    datas = sorted(datas, key=lambda x: mtx_order.index(x["matrix"]))

    # Calculate geo-mean speedups for BFS and Bellman-Ford
    bfs_speedups = calculate_speedups(datas, 'bfs')
    bellmanford_speedups = calculate_speedups(datas, 'bellmanford')

    # Output results for BFS
    print("Geometric Mean Speedups for BFS:")
    if bfs_speedups:
        for (baseline_method, comparison_method), geo_speedup in bfs_speedups.items():
            print(f"{baseline_method} vs {comparison_method}: {geo_speedup:.4f}")
    else:
        print("No valid BFS data for speedup calculation.")

    # Output results for Bellman-Ford
    print("\nGeometric Mean Speedups for Bellman-Ford:")
    if bellmanford_speedups:
        for (baseline_method, comparison_method), geo_speedup in bellmanford_speedups.items():
            print(f"{baseline_method} vs {comparison_method}: {geo_speedup:.4f}")
    else:
        print("No valid Bellman-Ford data for speedup calculation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate geometric speedup from JSON results.")
    parser.add_argument("filename", type=str, help="The path to the JSON file containing the results.")

    args = parser.parse_args()

    main(args.filename)

