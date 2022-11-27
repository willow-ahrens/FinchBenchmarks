#!/bin/bash

mkdir -p docker_results

docker create --name cgo23-finch-dummy cgo23-finch
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/spmspv_hb.json docker_results/spmspv_hb.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/triangle_results.json docker_results/triangle_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/conv_results.json docker_results/conv_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/alpha_results.json docker_results/alpha_results.json
docker cp cgo23-finch-dummy:/Finch-CGO-2023-Results/all_pairs.json docker_results/all_pairs.json
docker rm -f cgo23-finch-dummy