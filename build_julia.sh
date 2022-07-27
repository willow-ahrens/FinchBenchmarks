#!/bin/bash

# Build julia 1.7.2

export PATH=./julia:$PATH
export JULIA_PROJECT=.
export LD_LIBRARY_PATH=./taco/build/lib:$LD_LIBRARY_PATH

cd julia
make -j
cd ..
