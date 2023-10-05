# The Finch Benchmark Suite

## Installation

We use Julia to generate datasets. To install Julia, please follow
the instructions on the [Julia website](https://julialang.org/downloads/).

An example reference C++ implementation is provided in the `example/` directory.
All of the necessary files required to read a test dataset and benchmark the
implementation are provided as header-only dependencies. The only requirement is
a C++20 compliant compiler.

## Usage

Once Julia is installed, you can generate test data for a particular problem
with the `bin/generate.jl` script:

```bash
% julia bin/generate.jl --help
```

The script will generate a dataset for a particular problem and write it to
the given destination.

Implementations of a given kernel are expected to be given as executables that
take the same arguments as the reference implementation. All of the
implementations take the same arguments, namely:

```bash
% ./spmv --help
Usage: ./spmv [OPTIONS]
Options:
  -h, --help      Print this help message
  -i, --input     Specify the path for the inputs
  -o, --output    Specify the path for the outputs
  -v, --verbose   Print verbose output
```

Harnesses are provided to parse these arguments in several languages. To
benchmark your implementation in C++, simply include the `main` function given
in `src/benchmark.hpp`. This file includes the necessary header-only
dependencies to parse the arguments and read the input data, and calls the
`experiment` function, which is expected to benchmark the kernel.

## Formats

The SRBS uses the
[Binsparse](https://github.com/GraphBLAS/binsparse-specification) format to
store tensors on disk and in memory. In particular, the header of the file is a
JSON object that describes the dimensions and format of the tensor, and the data
for the tensor is stored as a set of named vectors. These objects may all be
stored in the same file, (e.g. [HDF5](https://www.hdfgroup.org/solutions/hdf5/)
or [ZARR](https://zarr.readthedocs.io/en/stable/)), or in separate files (e.g.
[.npy](https://numpy.org/doc/stable/reference/generated/numpy.load.html)).
