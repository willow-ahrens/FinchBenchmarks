#!/bin/bash
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400

cd /data/scratch/willow/FinchBenchmarks/spmv
source /afs/csail.mit.edu/u/w/willow/everyone/.bashrc

echo $SCRATCH
echo $JULIA_DEPOT_PATH
echo $JULIAUP_DEPOT_PATH
echo $PATH
echo $(pwd)
export TMPDIR=/tmp

# Use SLURM_ARRAY_TASK_ID for batch number (-b) and set the total number of batches (-B) to 20
julia run_spmv.jl -o spmv_results_3.json
