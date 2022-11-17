FROM docker.io/library/julia:1.8.2-bullseye 

RUN apt-get -y update 
RUN apt-get -y install cmake 
RUN apt-get -y install gcc
RUN apt-get -y install g++
RUN apt-get -y install python
RUN apt-get -y install python3
RUN apt-get -y install git

WORKDIR /Finch-CGO-2023-Results

COPY ./taco /Finch-CGO-2023-Results/taco
COPY ./taco-rle /Finch-CGO-2023-Results/taco-rle
COPY ./opencv /Finch-CGO-2023-Results/opencv

COPY spmv_taco.cpp /Finch-CGO-2023-Results/spmv_taco.cpp
COPY spmspv_taco.cpp /Finch-CGO-2023-Results/spmspv_taco.cpp
COPY alpha_taco_rle.cpp /Finch-CGO-2023-Results/alpha_taco_rle.cpp
COPY alpha_opencv.cpp /Finch-CGO-2023-Results/alpha_opencv.cpp
COPY triangle_taco.cpp /Finch-CGO-2023-Results/triangle_taco.cpp
COPY all_pairs_opencv.cpp /Finch-CGO-2023-Results/all_pairs_opencv.cpp
COPY conv_opencv.cpp /Finch-CGO-2023-Results/conv_opencv.cpp

COPY benchmark.hpp /Finch-CGO-2023-Results/benchmark.hpp
COPY npy.hpp /Finch-CGO-2023-Results/npy.hpp

COPY ./Makefile /Finch-CGO-2023-Results/Makefile
RUN make all

RUN mkdir -p /scratch

COPY ./Project.toml /Finch-CGO-2023-Results/
COPY ./Manifest.toml /Finch-CGO-2023-Results/
COPY ./Finch.jl /Finch-CGO-2023-Results/Finch.jl

COPY ./build_project.sh /Finch-CGO-2023-Results
RUN julia --project=/Finch-CGO-2023-Results -e "using Pkg; Pkg.instantiate()"

COPY ./spmspv.sl /Finch-CGO-2023-Results/spmspv.sl
COPY ./spmspv.jl /Finch-CGO-2023-Results/spmspv.jl
COPY ./TensorMarket.jl /Finch-CGO-2023-Results/TensorMarket.jl
RUN bash -e spmspv.sl
COPY ./spmspv_viz.jl /Finch-CGO-2023-Results/spmspv_viz.jl
# RUN julia --project=/Finch-CGO-2023-Results /Finch-CGO-2023-Results/spmspv_viz.jl

COPY ./triangle.sl /Finch-CGO-2023-Results/triangle.sl
COPY ./triangle.jl /Finch-CGO-2023-Results/triangle.jl
COPY ./triangle_viz.jl /Finch-CGO-2023-Results/triangle_viz.jl
RUN bash -e triangle.sl

COPY ./conv.sl /Finch-CGO-2023-Results/conv.sl
COPY ./conv.jl /Finch-CGO-2023-Results/conv.jl
COPY ./conv_viz.jl /Finch-CGO-2023-Results/conv_viz.jl
# RUN bash -e conv.sl

COPY ./alpha.sl /Finch-CGO-2023-Results/alpha.sl
COPY ./alpha.jl /Finch-CGO-2023-Results/alpha.jl
COPY ./alpha_viz.jl /Finch-CGO-2023-Results/alpha_viz.jl
RUN bash -e alpha.sl

COPY ./all_pairs.sl /Finch-CGO-2023-Results/all_pairs.sl
COPY ./all_pairs.jl /Finch-CGO-2023-Results/all_pairs.jl
COPY ./all_pairs_viz.jl /Finch-CGO-2023-Results/all_pairs_viz.jl
RUN bash -e all_pairs.sl

