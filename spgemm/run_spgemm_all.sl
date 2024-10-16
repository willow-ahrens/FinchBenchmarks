#!/bin/bash
#SBATCH --tasks-per-node=24
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400
#SBATCH --array=1-12%6

cd /data/scratch/willow/FinchBenchmarks/spgemm
source /afs/csail.mit.edu/u/w/willow/everyone/.bashrc

echo $SCRATCH
echo $JULIA_DEPOT_PATH
echo $JULIAUP_DEPOT_PATH
echo $PATH
echo $(pwd)
export TMPDIR=/tmp

julia run_spgemm.jl -d "small" --kernels "new" -b $SLURM_ARRAY_TASK_ID -B 12 -o lanka_mkl_small_$SLURM_ARRAY_TASK_ID.json
