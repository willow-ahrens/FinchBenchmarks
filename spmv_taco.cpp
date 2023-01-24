#include "taco/tensor.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <string>
#include "benchmark.hpp"

using namespace taco;

int main(int argc, char **argv) {
  if(argc != 4){
    std::cerr << "wrong number of arguments" << std::endl;
  }

  std::string file_y = argv[1];
  std::string file_A = argv[2];
  std::string file_x = argv[3];

  Tensor<double> y = read(file_y, Format({Dense}), true);
  Tensor<double> A = read(file_A, Format({Dense, Sparse}), true);
  Tensor<double> x = read(file_x, Format({Dense}), true);

  IndexVar i, j;

  y(i) += A(i, j) * x(j);

  // Compile the expression
  y.compile();
  y.assemble();

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&y]() {
      y.setNeedsCompute(true);
    },
    [&y]() {
      y.compute();
    }
  );

  std::cout << time << std::endl;

  write(file_y, y);

  return 0;
}