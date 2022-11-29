# Finch CGO 2023 Artifact

## Abstract

This artifact is packaged as a docker container which builds and runs all of the
experiments described in the paper. The container requires an x86 host, and runs
Debian bullseye. The artifact is implemented as a Julia package Finch.jl that
implements the looplets described in the paper, and a set of benchmark scripts
to run the benchmarks described in the paper. The specific runtimes measured
depend on the host computer, however we expect the relative performance between
Finch, TACO and OpenCV to roughly match the results presented in the paper.

## Artifact check-list (meta-information)
- **Compilation:**  This artifact compares to two versions of TACO which are
  included as source code and built. Additionally, we build OpenCV and several
  wrapper scripts to call taco.
- **Data set:** We use matrices from the Harwell-Boeing collection, graphs from
  the SNAP network dataset collection, and several image datasets, MNIST, EMNIST,
  Omniglot, and HumanSketches.
- **Hardware:** The artifact requires x86 processor.
- **Execution:** The experiments should be ran single threaded pinned to a
  single socket. The experiments from the paper take **TODO** in total. 
- **Metrics:** Execution time is reported.
- **Output:** The output is long format data including the execution time stored
  in JSON files. We also provide scripts to plot the data in the same format as
  the paper.
- **How much disk space required (approximately)?** **TODO**
- **Publicly available?:** Yes
- **Code licenses (if publicly available)?:** The code has been released under
  the MIT license. 
- **Archived?:** **TODO**

## Description
# How Delivered
Our artifact is distributed by cloning the repo from
[https://github.com/willow-ahrens/FinchBenchmarks.git](https://github.com/willow-ahrens/FinchBenchmarks.git).

# Installation

The artifact can be run containerized, using docker, or locally, using our
provided scripts.

If you choose to run containerized, Docker is used to build all the dependencies
and run the artifact. From the toplevel directory (e.g.
`.../FinchBenchmarks/`), run the `artifact.sh` script, which has two separate
commands. The first command runs docker to build, install, run, and plot the
results.  The dockerscript handles all parts of results collection and analysis.
The local build instructions contain more explanation of the steps the
Dockerscript takes.  The second command copies the results from the docker image
back into the toplevel directory on the host machine.   

To run locally, first install cmake, gcc, g++, python, python3, and git. Results
were originally collected using gcc 7.5.0, however a more modern version should
work. Next, run the `build_project.sh` script to download julia 1.8.2 and build
OpenCV and TACO (this uses the included Makefile). The results can be collected
using the `run.sh` script which runs all of the experiments and generates the
graphs shown in the paper. Running the data collection scripts automatically
downloads the appropriate datasets.

TODO however they are also included in the artifact distribution. 

# Experiment Workflow
There are five experiments described in the paper in sections 9.1 through 9.5,
and each has an associated bash script to run the experiment and generate the
associated graphs, and a julia script to collect the benchmark data. There are
separate binaries for the OpenCV and TACO experiments, which are called from the
Julia files with the appropriate environment variables set. Each section is
described as follows.

- SpMSpV: 
- Triangle Counting: 
- Convolution: 
- Alpha Blending: 
- All-Pairs Image Similarity:  

The JSON files produced contain the **TODO: describe format of the output json**. 

**TODO: describe high level implementation structure of Finch?**

# Evaluation and expected result
Running the artifact as described above produces the raw execution timing, and
the relative performance plotted in graphs which match the paper. We include
reference results which match the paper, and the scripts compare the relative
speedups of the artifact to the reference. **TODO: More
details here**.

# Experiment customization

It is possible to customize the benchmark scripts to use additional datasets.
The matrices used in SpMSpV and Triangle counting are downloaded using the
MatrixDepot Julia package. In `spmspv.jl`, the `hb` variable can be modified to
use any other matrix from MatrixDepot. Similarly, in `triangle.jl`, the main
function can be modified by adding an additional matrix name to the list of
tests. 

The `alpha.jl` and `all_pairs.jl` datasets use images downloaded using the
TensorDepot Julia package. Other datasets included in this package can be used,
as long as they are a 3-tensor with the first index used as the index of the
image, and the next two indices represent the rows and column of the image. The
`permutedims` function can be used to permute the dimensions if they do not
match. Other datasets can be added to TensorDepot for easy integration into the
test harness, or they can be downloaded directly. 

The experiments can be run on different platforms with Julia support, by cloning
the repository
[https://github.com/willow-ahrens/FinchBenchmarks](https://github.com/willow-ahrens/FinchBenchmarks). 

Finch can also be used as a standalone sparse tensor compiler. More details and
documentation is available at
[https://github.com/willow-ahrens/Finch.jl](https://github.com/willow-ahrens/Finch.jl). 