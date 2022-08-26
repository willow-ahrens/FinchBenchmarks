#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 240:00:00
#SBATCH -t 4-0
#SBATCH -e slurm-%A_%a.err
#SBATCH -o slurm-%A_%a.out
#SBATCH --partition=lanka-v3
#SBATCH --nodelist lanka[26,27,28]

../julia/julia --project=. triangle.jl triangle_results.json