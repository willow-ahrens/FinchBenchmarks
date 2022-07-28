#!/bin/bash

mkdir -p opencv/build
mkdir -p opencv/install
cd opencv/build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../install -DBUILD_ZLIB=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF .. 
make -j$(nproc)
make install