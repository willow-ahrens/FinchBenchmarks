#!/bin/bash
#SBATCH --tasks-per-node=24
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 12:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400

cd /data/scratch/willow/FinchBenchmarks/spgemm
source /afs/csail.mit.edu/u/w/willow/everyone/.bashrc

echo $SCRATCH
echo $JULIA_DEPOT_PATH
echo $JULIAUP_DEPOT_PATH
echo $PATH
echo $(pwd)
export TMPDIR=/tmp

# Call the Julia function with the selected dataset and output file
julia run_spgemm.jl -d "joel_sm" -o "lanka_sm.json"
julia run_spgemm.jl -d "joel_lg2" -o "lanka_lg2.json"
