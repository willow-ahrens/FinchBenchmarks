#!/bin/bash
#SBATCH --tasks-per-node=1
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH -t 04:00:00
#SBATCH --partition=lanka-v3
#SBATCH --qos=commit-main
#SBATCH --mem 102400
#SBATCH --array=[4,8]

cd /data/scratch/willow/FinchBenchmarks/morphology
source /afs/csail.mit.edu/u/w/willow/everyone/.bashrc

echo $SCRATCH
echo $JULIA_DEPOT_PATH
echo $JULIAUP_DEPOT_PATH
echo $PATH
echo $(pwd)

# Define an array of datasets
DATASETS=("mnist" "omniglot" "humansketches" "testimage_dip3e" "mnist_magnify" "omniglot_magnify" "humansketches_magnify" "testimage_dip3e_magnify")
# Get the corresponding dataset for this job index
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID - 1]}

julia +1.9 run_morphology.jl --dataset $DATASET --output morphology_results_$SLURM_ARRAY_TASK_ID.json --num_trials 100
