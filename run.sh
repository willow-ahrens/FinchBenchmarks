
# Initialize environment variables

export PATH=./julia:$PATH
export JULIA_PROJECT=.
export LD_LIBRARY_PATH=./taco/build/lib:$LD_LIBRARY_PATH
export DYLD_FALLBACK_LIBRARY_PATH=./taco/build/lib:$DYLD_FALLBACK_LIBRARY_PATH

#julia spmv.jl
#julia spgemm.jl
#julia spmv2.jl
#julia spgemm2.jl
#julia spgemmh.jl
#julia smttkrp.jl
