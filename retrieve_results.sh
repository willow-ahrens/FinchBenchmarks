#!/bin/bash

docker create --name cgo-finch23-dummy cgo-finch23
docker cp cgo-finch23-dummy:/Finch-CGO-2023-Results/alpha.json docker_results/alpha.json

docker rm -f cgo-finch23-dummy