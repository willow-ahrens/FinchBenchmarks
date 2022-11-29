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
COPY ./TensorDepot.jl /Finch-CGO-2023-Results/TensorDepot.jl

COPY ./build_project.sh /Finch-CGO-2023-Results
RUN julia --project=/Finch-CGO-2023-Results -e "using Pkg; Pkg.instantiate()"
COPY ./TensorMarket.jl /Finch-CGO-2023-Results/TensorMarket.jl

COPY ./alpha.sh /Finch-CGO-2023-Results/alpha.sh
COPY ./alpha.jl /Finch-CGO-2023-Results/alpha.jl
RUN bash -e alpha.sh

COPY ./all_pairs.sh /Finch-CGO-2023-Results/all_pairs.sh
COPY ./all_pairs.jl /Finch-CGO-2023-Results/all_pairs.jl
RUN bash -e all_pairs.sh

COPY ./spmspv.sh /Finch-CGO-2023-Results/spmspv.sh
COPY ./spmspv.jl /Finch-CGO-2023-Results/spmspv.jl
RUN bash -e spmspv.sh

COPY ./triangle.sh /Finch-CGO-2023-Results/triangle.sh
COPY ./triangle.jl /Finch-CGO-2023-Results/triangle.jl
RUN bash -e triangle.sh

COPY ./conv.sh /Finch-CGO-2023-Results/conv.sh
COPY ./conv.jl /Finch-CGO-2023-Results/conv.jl
RUN bash -e conv.sh

COPY ./spmspv_plot.sh /Finch-CGO-2023-Results/spmspv_plot.sh
COPY ./spmspv_plot.jl /Finch-CGO-2023-Results/spmspv_plot.jl
RUN bash -e spmspv_plot.sh

COPY ./triangle_plot.sh /Finch-CGO-2023-Results/triangle_plot.sh
COPY ./triangle_plot.jl /Finch-CGO-2023-Results/triangle_plot.jl
RUN bash -e triangle_plot.sh

COPY ./conv_plot.sh /Finch-CGO-2023-Results/conv_plot.sh
COPY ./conv_plot.jl /Finch-CGO-2023-Results/conv_plot.jl
RUN bash -e conv_plot.sh

COPY ./alpha_plot.sh /Finch-CGO-2023-Results/alpha_plot.sh
COPY ./alpha_plot.jl /Finch-CGO-2023-Results/alpha_plot.jl
RUN bash -e alpha_plot.sh

COPY ./all_pairs_plot.sh /Finch-CGO-2023-Results/all_pairs_plot.sh
COPY ./all_pairs_plot.jl /Finch-CGO-2023-Results/all_pairs_plot.jl
RUN bash -e all_pairs_plot.sh
