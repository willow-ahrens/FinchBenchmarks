#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 240:00:00
#SBATCH -t 4-0
#SBATCH -e slurm-%A_%a.err
#SBATCH -o slurm-%A_%a.out

export SCRATCH=/data/scratch/pahrens
export PATH="$SCRATCH/julia:$PATH"
export JULIA_DEPOT_PATH=/data/scratch/pahrens/.julia
export MATRIXDEPOT_DATA=/data/scratch/pahrens/MatrixData

../julia/julia --project=. conv.jl conv_results_results.json
