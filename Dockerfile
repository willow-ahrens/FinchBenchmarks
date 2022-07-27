FROM docker.io/library/julia:1.7.2-bullseye

RUN apt-get -y update 
RUN apt-get -y install cmake 
RUN apt-get -y install gcc
RUN apt-get -y install g++
RUN apt-get -y install python
RUN apt-get -y install python3
RUN apt-get -y install git

WORKDIR Finch-CGO-2023-Results

COPY ./taco /Finch-CGO-2023-Results/taco
COPY ./build_taco.sh /Finch-CGO-2023-Results
RUN bash -e build_taco.sh

COPY ./Project.toml /Finch-CGO-2023-Results/
COPY ./Manifest.toml /Finch-CGO-2023-Results/
COPY ./Pigeon.jl /Finch-CGO-2023-Results/Pigeon.jl
COPY ./build_project.sh /Finch-CGO-2023-Results
RUN bash -e build_project.sh

#COPY ./julia /Finch-CGO-2023-Results/julia
#COPY ./build_julia.sh /Finch-CGO-2023-Results
#RUN bash -e build_julia.sh

#COPY ./paper.jl /Finch-CGO-2023-Results
#COPY ./spmv.jl /Finch-CGO-2023-Results
#COPY ./spgemm.jl /Finch-CGO-2023-Results
#COPY ./spmv2.jl /Finch-CGO-2023-Results
#COPY ./run.sh /Finch-CGO-2023-Results
#
RUN bash -e run.sh

COPY . /Finch-CGO-2023-Results

RUN bash -e analysis.sh