#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
bash -e $SCRIPT_DIR/download_julia.sh
source $SCRIPT_DIR/julia_env.sh

mkdir -p $SCRIPT_DIR/scratch
mkdir -p $SCRIPT_DIR/scratch/MatrixData
mkdir -p $SCRIPT_DIR/scratch/TensorData
make all
julia --project=$SCRIPT_DIR -e "using Pkg; Pkg.instantiate()"