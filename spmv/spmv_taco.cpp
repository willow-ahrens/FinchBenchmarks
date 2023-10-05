#include "taco.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

namespace fs = std::__fs::filesystem;

using namespace taco;

void experiment(std::string input, std::string output, int verbose){
    Tensor<double> A = read(fs::path(input)/"A.ttx", Format({Dense, Sparse}), true);
    Tensor<double> x = read(fs::path(input)/"x.ttx", Format({Dense}), true);
    int m = A.getDimension(0);
    int n = A.getDimension(1);
    Tensor<double> y("y", {n}, Format({Dense}));

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

    write("y.ttx", y);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(fs::path(output)/"measurements.json");
    measurements_file << measurements;
    measurements_file.close();
}