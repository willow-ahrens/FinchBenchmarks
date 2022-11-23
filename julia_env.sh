SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export JULIA_DEPOT_PATH=$SCRIPT_DIR/.julia
export PATH=$PATH:$SCRIPT_DIR/julia-1.8.2/bin