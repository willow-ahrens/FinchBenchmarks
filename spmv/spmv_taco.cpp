#include "taco.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::filesystem;

using namespace taco;

int main(int argc, char **argv){
    auto params = parse(argc, argv);
    Tensor<double> A = read(fs::path(params.input)/"A.ttx", Format({Dense, Sparse}), true);
    Tensor<double> x = read(fs::path(params.input)/"x.ttx", Format({Dense}), true);
    int m = A.getDimension(0);
    int n = A.getDimension(1);
    Tensor<double> y("y", {m}, Format({Dense}));

    IndexVar i, j;

    y(i) += A(i, j) * x(j);

    //perform an spmv of the matrix in c++

    y.compile();

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
      [&y]() {
        y.setNeedsAssemble(true);
        y.setNeedsCompute(true);
      },
      [&y]() {
        y.assemble();
        y.compute();
      }
    );

    write(fs::path(params.input)/"y.ttx", y);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(fs::path(params.output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}
