#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 240:00:00
#SBATCH -t 4-0
#SBATCH -e slurm-%A_%a.err
#SBATCH -o slurm-%A_%a.out
#SBATCH --partition=lanka-v3

if [ -n $SLURM_JOB_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_DIR=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
    SCRIPT_DIR=$(dirname $SCRIPT_DIR)
else
    SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
fi

export SCRATCH=$SCRIPT_DIR/scratch
export MATRIXDEPOT_DATA=$SCRATCH/MatrixData
export TENSORDEPOT_DATA=$SCRATCH/TensorData
export DATADEPS_ALWAYS_ACCEPT=true

bash -e $SCRIPT_DIR/download_julia.sh
source julia_env.sh

julia --project=. spmspv.jl spmspv_hb.json
