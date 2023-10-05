#!/bin/bash

export JULIA_VERSION=1.8.2
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# echo $PATH
# echo $JULIA_DEPOT_PATH

# if [[ "$(command -v julia)" == "" || "$(julia --version)" != "julia version $JULIA_VERSION" ]]
# then 
#     echo "INCORRECT or MISSING"
cd $SCRIPT_DIR
if [[ -d $SCRIPT_DIR/julia-$JULIA_VERSION ]]
then 
echo "We have already downloaded julia"
    export JULIA_DEPOT_PATH=$SCRIPT_DIR/.julia
    mkdir -p $JULIA_DEPOT_PATH
else 
    echo "Downloading julia"
    if [[ "$(uname)" == "Darwin" ]] 
    then
        wget https://julialang-s3.julialang.org/bin/mac/x64/1.8/julia-$JULIA_VERSION-mac64.tar.gz
        tar xvzf julia-$JULIA_VERSION-mac64.tar.gz
    else 
        wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-$JULIA_VERSION-linux-x86_64.tar.gz
        tar xvzf julia-$JULIA_VERSION-linux-x86_64.tar.gz
    fi
    export JULIA_DEPOT_PATH=$SCRIPT_DIR/.julia
    mkdir -p $JULIA_DEPOT_PATH
fi
export PATH=$PATH:$SCRIPT_DIR/julia-$JULIA_VERSION/bin
# else 
#     echo "Already Present"
# fi

# echo $PATH
# echo $JULIA_DEPOT_PATH