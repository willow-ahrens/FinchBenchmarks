#!/bin/bash

if [ -z "$NPROC_VAL" ]
then
    NPROC_VAL=$(nproc)
fi

mkdir -p opencv/build
mkdir -p opencv/install
cd opencv/build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../install -DBUILD_ZLIB=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_PNG=ON .. 
make -j$NPROC_VAL
make install