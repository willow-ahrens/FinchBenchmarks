#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d Finch.jl ] || git clone https://github.com/finch-tensor/Finch.jl Finch.jl
git -C Finch.jl fetch origin --tags wma/split-offset-merged

[ -d Finch-wingspan.jl ] || git -C Finch.jl worktree add ../Finch-wingspan.jl wingspan-arxiv-v0
[ -d Finch-birdseed.jl ] || git -C Finch.jl worktree add ../Finch-birdseed.jl origin/wma/split-offset-merged

SIF_IMAGE="${SIF_IMAGE:-$PWD/wingspan_cgo27.sif}"
mkdir -p .julia-depot
rm -f envs/*/Manifest.toml  # regenerate under the container's Julia

apptainer exec --cleanenv --no-home \
    --bind "$PWD:/repo" \
    --env JULIA_DEPOT_PATH=/repo/.julia-depot \
    "$SIF_IMAGE" bash -c '
cd /repo
for e in wingspan birdseed; do
  julia --project=envs/$e -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
done'
