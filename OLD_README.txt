Welcome Artifact Evaluator!

#--------#
| Claims |
#--------#

We claim the results for spmv2 and spgemm in figures 6 and 7 are replicable
(under reasonable expectations of performance variation).

Due to time constraints, we have chosen not to ask artifact evaluators to
reproduce results for the kernels spmv, spgemm2, spgemmh, or spmttkrp.

#-----------------------#
| Getting Started Guide |
#-----------------------#

1. Many scripts will need to be run from the directory that contains this
README.  For your convenience, we have included a Dockerfile to copy this
directory to a container and build and run the required project dependencies.
If it's at all possible for you to execute 

    docker build -t pigeon .

from the project directory, you can skip to step 5. This command took about an
hour to run on our machine.

2. If you'd rather not use the Dockerfile, there are three main dependencies to
the project that you must build. Each dependency is included in it's own
directory in the artifact, and we expect them to be built locally.

    - `julia` 1.7.2 is our chosen version of the julia programming language.
    You can build by running the `build_julia.sh` script from the toplevel
    directory.  Julia contains a README explaining build steps in detail. Add
    `julia/julia` to the `PATH`.
        
    - `taco` is the project that writes the tensor expressions we autotune. We
    have included a special version of taco that contains some interface
    modifications needed to replicate our results. You can build by running the
    `build_taco.sh` script from the toplevel directory. The taco directory
    contains a README explaining build steps in detail. You'll need to install
    `cmake` for this step. Add `taco/build/lib` to the `LD_LIBRARY_PATH`.

    - `Pigeon.jl` is the julia project which contains our autotuner
    implementation.  You can download project dependencies by running
    the `build_project.sh` script from the toplevel directory.  For those
    familiar with Julia, the project dependencies for all the julia scripts are
    contained in the `Project.toml` and `Manifest.toml` files in the toplevel
    directory, including a dependency on the locally developed package
    `Pigeon.jl`. Set `JULIA_PROJECT` to the location of the toplevel directory.
    If you'd like, set `JULIA_DEPOT_PATH` to a good place to put julia project
    dependencies.

#---------------------------#
| Step-by-Step Instructions |
#---------------------------#

3. Run the `run.sh` script from the toplevel directory to autotune and benchmark
a subset of our test kernels.

4. Run the `analysis.sh` script to interpret the results and compare to Figures 6
and 7 in the paper.

5. Evaluate our claims. Our paper contains experiments with 6 test kernels,
named spmv, spmv2, spgemm, spgemm2, spgemmh, and spmttkrp. They are described in
Figure 6. Spmv is a trivial case, and the results for spgemm2, spgemmh, and
spmttkrp each take a very long time to collect (several hours).  Therefore, we
only expect the artifact evaluation to consider the results for the two kernels
spmv2 and spgemm. The other kernels are optional.  These two kernels are fairly
representative of the full space of scheduling decisions. Spmv2 requires an
intermediate tensor to minimize loop depth, and spgemm requires a specific loop
ordering for appropriate iteration space filtering.

All of our results center around Figures 6 and 7. The code to collect all the
data for a particular kernel is included in a `*.jl` file named after the
kernel, so running 

    julia --project=. spmv.jl

will collect data for the spmv kernel. Data for the two core kernels should have
been collected at this point by the Dockerfile or from running `run.sh`.
Our scripts will generate a `.json` file with the same name to hold the data.
The `.json` files we produced are included in the `reference_results` folder. A
guide to interpreting our `.json` files is included in step 7. Our code also
produces some `.bson` files, which can be ignored.

The `analysis.jl` script will gather all `*.json` files passed to it and
visualize the results as they are presented in the paper.  A table for figure 6
will be written to the file "figure6.txt", and tables for figure 7 will be
written to the file "figure7.txt".  Additionally, the table for figure 6 and
unicode plots for figure7 will be output to the command line by the
`analysis.jl` script.  You can compare the output of these scripts to the
contents of the paper. Any counts of schedules should match exactly, but
performance variations might be noticed. Since our autotuner selects a kernel
from the asymptotic frontier using empirical measurements, the autotuner might
select an asymptotically different kernel than the one we used for the more
complicated kernels. However, the asymptotic frontiers for spmv2 and spgemm are
quite homogeneous, so we expect that if our plots show an obvious big speedup of
the autotuned kernels over the default ones, that should be reflected in the
evaluator's results.

6. Begin inspecting the built container with

    docker run --name session --rm -t -i pigeon bash

If you would like to run the analysis script again to see its output, try

    julia --project=. analysis.jl *.json

7. [Optional] We have included the data that generated the results from our
paper in the `reference_results` directory, should you wish to compare directly.
If you wanted to compare against them, try

    cd reference_results
    julia --project=.. ../analysis.jl *.json
    cd ..

The names of the fields in each `.json` file are as follows:

N: The default dimension of square tensors used in the kernel. This is set by
measuring the runtime of the default kernel on your machine, it may not match
the values in our paper.

universe_build_time: The time required to enumerate all minimum depth schedules.
universe_length: The number of minimum depth schedules.
frontier_filter_time: The time required to filter out the asymptotically
    undominated frontier of minimum depth schedules.
frontier_length: The size of the asymptotically undominated frontier of minimum
    depth schedules.

tacoverse_build_time: The time required to enumerate all minimum depth
    taco-compatible schedules.
tacoverse_length: The number of minimum depth taco-compatible schedules.
tacotier_filter_time: The time required to filter out the asymptotically
    undominated frontier of minimum depth taco-compatible schedules.
tacotier_length: The size of the asymptotically undominated frontier of minimum
    depth taco-compatible schedules.

tacotier_bench: The runtime of each kernel in the taco frontier on uniformly
    distributed square inputs of dimension N and density 0.01.
default_kernel_bench: The runtime of the default kernel on the same inputs.

n_series: A series of square dimension sizes for tensors of density 0.01.
default_n_series: The runtime of the default kernel on these inputs.
default_n_series: The runtime of the autotuned kernel on these inputs.

p_series: A series of densities for tensors of size N. We have truncated this
range to reduce the probability of taco segfaults at extremely low tensor
density on some systems.
default_p_series: The runtime of the default kernel on these inputs.
default_p_series: The runtime of the autotuned kernel on these inputs.

8. [optional] If you'd like to take a look at the code which generates these
results, you can start with the scripts themselves (e.g. spmv.jl) which set up
the kernels to be benchmarked, and the file `paper.jl`. The file `paper.jl` does
most of the test harnessing, and makes several calls to functions in our 
autotuning infrastructure, `Pigeon.jl`.