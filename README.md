# Finch CGO 2023 Artifact

## Abstract

This artifact has two scripts that build and run all of the experiments
described in the paper. We require an x86 host, and runs Debian bullseye. The
artifact is implemented as a Julia package Finch.jl that implements the looplets
described in the paper, and a set of benchmark scripts to run the benchmarks
described in the paper. The specific runtimes measured depend on the host
computer, however we expect the relative performance between Finch, TACO and
OpenCV to roughly match the results presented in the paper.

## Artifact check-list (meta-information)
- **Operating System:** We tested on Ubuntu 18 and MacOS 12.5, but we expect it
to work on most modern mac or linux distributions.
- **Compilation:**  This artifact compares to two versions of TACO which are
  included as source code and built. Additionally, we build OpenCV and several
  wrapper scripts to call taco.
- **Data set:** We use matrices from the Harwell-Boeing collection, graphs from
  the SNAP network dataset collection, and several image datasets, MNIST, EMNIST,
  Omniglot, and HumanSketches (these are automatically downloaded).
- **Software:** The artifact requires cmake (we used 3.10), gcc/g++ (we used
  7.5), Python (we used 2.7/3.9), and git (we used 2.17).  We used Julia 1.8.2
  (which is automatically downloaded). We have several Julia dependencies whose
  exact versions are recorded in the Manifest.toml file (these are also
  automatically downloaded).
- **Hardware:** The artifact requires x86 processor.
- **Execution:** The experiments should be ran single threaded pinned to a
  single socket. The experiments from the paper take 2 hours in total. 
- **Metrics:** Execution time is reported.
- **Output:** The output is long format data including the execution time stored
  in JSON files. We also provide scripts to plot the data in the same format as
  the paper.
- **How much disk space required (approximately)?** 4GB
- **Publicly available?:** Yes
- **Code licenses (if publicly available)?:** The code has been released under
  the MIT license. 
- **Archived?:** (Yes)[https://doi.org/10.5281/zenodo.7499790].
All of the datasets we used are publicly accessible with considerations for
access in perpetuity.

# How Delivered
Our artifact is distributed by (direct
download)[https://doi.org/10.5281/zenodo.7499790] or by cloning the repo from
[github](https://github.com/willow-ahrens/FinchBenchmarks/tree/cgo23-artifact).
using the following command:

```
git clone -b cgo23-artifact https://github.com/willow-ahrens/FinchBenchmarks.git
```

Notice that we use the cgo23-artifact branch of the repo.

# Installation

1. First install cmake, gcc, g++, Python, Python3, and git. Results were
originally collected using gcc 7.5.0, however a more modern version should work.
The gcc compiler only affects TACO and OpenCV, not Finch.

2. Run the `build.sh` script to download Julia 1.8.2 and build
OpenCV and TACO (this uses the included Makefile).


# Experiment Workflow
There are five experiments described in the paper in sections 9.1 through 9.5,
with associated scripts to collect and analyze results. You can run all these
commands with the `run.sh` script. This script runs all of the experiments and
generates the graphs shown in the paper. Running the data collection scripts
automatically downloads the appropriate datasets.

The experiments are named as follows:
  - `all_pairs` (for all-pairs image similarity)
  - `alpha` (for alpha blending)
  - `conv` (for sparse convolution)
  - `smpspv` (for sparse matrix sparse vector multiply)
  - `triangle` (for triangle counting)

Each experiment has several associated scripts that are all prefixed by it's
name. We use `alpha` as an example.

`alpha.jl` is a Julia script that runs the experiments. It can be invoked as

`julia --project=. alpha.jl RESULT_FILE`

where `RESULT_FILE` is the name of the json output in long format.

`alpha.sh` is a bash script that runs `alpha.jl` after setting appropriate
environment variables to keep dataset and Julia package downloads inside the
toplevel directory. This script produces `alpha_results.json`, the results.

`alpha_plot.jl` is a Julia script that reads the results file and generates
plots. It can be invoked as 

`julia --project=. alpha_plot.jl RESULT_FILE PLOT_FILE`

where `PLOT_FILE` is the name of the output plot (with a `.png` extension).

`alpha_plot.sh` is a bash script that runs `alpha.jl` after setting similar
appropriate environment variables. This script produces `alpha_plot.png`,
the plot of the results you just collected.

`alpha_results_reference.json` contains the results we used to generate the
plots for the paper. You can point the plotting scripts at this file to
reproduce the results in the paper exactly (you could also just look at the
plots in the paper).

There are separate binaries for the OpenCV and TACO experiments, which are
called from the Julia files with the appropriate environment variables set. 

## Short lists

To save time, `spmspv.jl` and `triangle.jl` run a subset of results in the paper
consisting of the 5 smallest and 5 best performing datasets. To run the full
versions, pass `long` as an additional final argument to these scripts.

# Evaluation and expected result
Running the artifact as described above produces the raw execution timing, and
the relative performance plotted in graphs which match the paper. You can verify
that each experiment matches it's corresponding figure in the paper. We also
include reference results which were used to generate the figures in the paper.

# Experiment customization

It is possible to customize the benchmark scripts to use additional datasets.
The matrices used in SpMSpV and Triangle counting are downloaded using the
MatrixDepot Julia package. The main functions of these scripts can be modified
to use a different list of matrix names recognized by
[MatrixDepot](https://github.com/JuliaLinearAlgebra/MatrixDepot.jl).

The `alpha.jl` and `all_pairs.jl` datasets use images downloaded using the
TensorDepot Julia package. Other datasets included in this package can be used,
as long as they are a 3-tensor with the first index used as the index of the
image, and the next two indices represent the rows and column of the image. The
`permutedims` function can be used to permute the dimensions if they do not
match. Other datasets can be added to TensorDepot for easy integration into the
test harness, or they can be downloaded directly. 

Finch can also be used as a standalone sparse tensor compiler. More details and
documentation is available at
[https://github.com/willow-ahrens/Finch.jl](https://github.com/willow-ahrens/Finch.jl).
Very briefly, any `@finch` macro expression in the benchmark suite is calling
the Finch compiler and using Looplets to compile sparse kernels. The `@fiber`
macro sets up level formats with the abbreviations of `d` for a dense level,
`sl` for a sparse list of nonzeros, `sv` for a VBL level, and `rl` for a
RLE level.
