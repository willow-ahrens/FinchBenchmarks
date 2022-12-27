#!/bin/bash

export SCRATCH=$(pwd)/scratch
mkdir -p $SCRATCH
export MATRIXDEPOT_DATA=$SCRATCH/MatrixData
export TENSORDEPOT_DATA=$SCRATCH/TensorData
export DATADEPS_ALWAYS_ACCEPT=true
export DATADEPS_NO_STANDARD_LOAD_PATH=true
export DATADEPS_DISABLE_ERROR_CLEANUP=true

export SCRIPT_DIR=$(pwd)
if [[ -f "$SCRIPT_DIR/download_julia.sh" ]]; then
    bash -e $SCRIPT_DIR/download_julia.sh
    source julia_env.sh
fi

julia --project=. alpha_plot.jl alpha_results.json alpha_plot.png