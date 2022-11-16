#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 240:00:00
#SBATCH -t 4-0
#SBATCH -e slurm-%A_%a.err
#SBATCH -o slurm-%A_%a.out

export SCRATCH=/SCRATCH
export MATRIXDEPOT_DATA=/$SCRATCH/MatrixData

julia --project=. conv.jl conv_results.json
