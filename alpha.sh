#!/bin/bash

if [[ -n $SCRIPT_DIR ]];  then
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
fi

export SCRATCH=$SCRIPT_DIR/scratch
export MATRIXDEPOT_DATA=$SCRATCH/MatrixData
export TENSORDEPOT_DATA=$SCRATCH/TensorData
export DATADEPS_ALWAYS_ACCEPT=true

if [[ -f "$SCRIPT_DIR/download_julia.sh" ]]; then
    bash -e $SCRIPT_DIR/download_julia.sh
    source julia_env.sh
fi

julia --project=. alpha.jl alpha_results.json
