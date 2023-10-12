#include "taco.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

//namespace fs = std::__fs::filesystem;

using namespace taco;

void experiment(std::string input, std::string output, int verbose){
    Tensor<double> A = read(input+".ttx", Format({Dense, Sparse}), true);
    Tensor<double> B = read(input+"_s.ttx", Format({Dense, Sparse}), true);
    int m = A.getDimension(0);
    int mn = A.getDimension(1);
    int n = B.getDimension(1);
    Tensor<double> C("C", {m, n}, Format({Dense, Sparse}));

    IndexVar i, j, k;

    C(i, j) += A(i, k) * B(k, j);

    //perform an spmv of the matrix in c++

    C.compile();

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
      [&C]() {
        C.setNeedsAssemble(true);
        C.setNeedsCompute(true);
      },
      [&C]() {
        C.assemble();
        C.compute();
      }
    );

    //write("C.ttx", C);
C.printComputeIR(std::cout, true, true);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(output+".json");
    measurements_file << measurements;
    measurements_file.close();
}

/*
void experiment(std::string input, std::string output, int verbose){
    Tensor<double> A = read(input+"/A.ttx", Format({Dense, Sparse}), true);
    Tensor<double> x = read(input+"/x.ttx", Format({Dense}), true);
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
    std::ofstream measurements_file(output+"/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
}*/

