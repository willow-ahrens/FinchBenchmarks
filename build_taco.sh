#!/bin/bash

# Build our special version of taco

cd taco
mkdir build
cd build
cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release ..
make -j8
cd ../..
