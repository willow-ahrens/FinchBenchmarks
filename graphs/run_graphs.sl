#!/bin/bash
#SBATCH --tasks-per-node=24
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400
#SBATCH --array=1-6%6

cd /data/scratch/willow/FinchBenchmarks/graphs
source /afs/csail.mit.edu/u/w/willow/everyone/.bashrc

echo $SCRATCH
echo $JULIA_DEPOT_PATH
echo $JULIAUP_DEPOT_PATH
echo $PATH
echo $(pwd)

# Call the Julia function with the selected dataset and output file
julia run_graphs.jl -d "yang${SLURM_ARRAY_TASK_ID}" -o "graphs_data_${SLURM_ARRAY_TASK_ID}.json"

