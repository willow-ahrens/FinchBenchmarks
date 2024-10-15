#!/bin/bash

for t in {1..4}
do
    echo "Running run_spadd.jl with $t threads"
    
    # Run Julia command with current number of threads
    julia --threads="$t" "run_spadd.jl" -a
done
