
#!/bin/bash

# Install julia dependencies

export PATH=./julia:$PATH
export JULIA_PROJECT=.

julia -e "using Pkg; Pkg.Registry.add(\"General\"); Pkg.resolve(); Pkg.instantiate()"
