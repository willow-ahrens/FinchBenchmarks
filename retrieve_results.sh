#!/bin/bash

mkdir -p docker_results

docker create --name cgo23-finch-dummy cgo23-finch
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/spmspv_results.json docker_results/spmspv_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/triangle_results.json docker_results/triangle_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/conv_results.json docker_results/conv_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/alpha_results.json docker_results/alpha_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/all_pairs_results.json docker_results/all_pairs_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/spmspv_plot_1dense.png docker_results/spmspv_plot_1dense.png
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/spmspv_plot_10count.png docker_results/spmspv_plot_10count.png
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/triangle_plot.png docker_results/triangle_plot.png
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/conv_plot.png docker_results/conv_plot.png
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/alpha_plot.png docker_results/alpha_plot.png
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/all_pairs_plot.png docker_results/all_pairs_plot.png
docker rm -f cgo23-finch-dummy