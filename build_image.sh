#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"

SIF_IMAGE="${SIF_IMAGE:-$REPO_ROOT/wingspan_cgo27.sif}"
DOCKER_REF="${DOCKER_REF:-docker://agushin/wingspan-cgo27-artifact:v1}"

if [ -f "$SIF_IMAGE" ]; then
    if [ "${FORCE_PULL:-0}" = "1" ]; then
        rm -f "$SIF_IMAGE"
    fi
fi

if [ ! -f "$SIF_IMAGE" ]; then
    echo "Pulling $DOCKER_REF -> $SIF_IMAGE"
    apptainer pull "$SIF_IMAGE" "$DOCKER_REF"
fi

FINCH_URL=https://github.com/finch-tensor/Finch.jl.git
[ -d "$REPO_ROOT/Finch-wingspan.jl" ] || git clone --branch wingspan-arxiv-v0 "$FINCH_URL" "$REPO_ROOT/Finch-wingspan.jl"
[ -d "$REPO_ROOT/Finch-birdseed.jl" ] || git clone --branch wma/split-offset-merged "$FINCH_URL" "$REPO_ROOT/Finch-birdseed.jl"

mkdir -p "$REPO_ROOT/.julia-depot" "$REPO_ROOT/.poetry-venv"

apptainer exec --cleanenv --no-home \
    --bind "$REPO_ROOT:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    --env POETRY_VIRTUALENVS_PATH=/repo/.poetry-venv \
    "$SIF_IMAGE" bash -c '
set -e
cd /repo
make all
for p in . envs/wingspan envs/birdseed; do
    julia --project=$p -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
done
poetry install --no-root
'

echo "Done."
