#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 240:00:00
#SBATCH -t 4-0
#SBATCH -e slurm-%A_%a.err
#SBATCH -o slurm-%A_%a.out
#SBATCH --partition=lanka-v3

export SCRATCH=/SCRATCH
export MATRIXDEPOT_DATA=/$SCRATCH/MatrixData
export DATADEPS_ALWAYS_ACCEPT=true

julia --project=. all_pairs.jl all_pairs.json
