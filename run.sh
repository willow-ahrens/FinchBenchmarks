#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 240:00:00
#SBATCH -t 4-0
#SBATCH -e slurm-%A_%a.err
#SBATCH -o slurm-%A_%a.out
#SBATCH --partition=lanka-v3

bash -e spmspv.sl
bash -e triangle.sl
bash -e conv.sl
bash -e alpha.sl 
bash -e all_pairs.sl