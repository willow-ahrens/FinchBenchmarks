#!/bin/bash

# Build our special version of taco

TACO_DIR=${1:-taco}

cd $TACO_DIR
mkdir -p build
cd build
cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../..
