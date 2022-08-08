#!/bin/bash

# Build our special version of taco

TACO_DIR=${1:-taco}

if [ -z "$NPROC_VAL" ]
then
    NPROC_VAL=$(nproc)
fi

cd $TACO_DIR
mkdir -p build
cd build
cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release ..
make -j$NPROC_VAL
