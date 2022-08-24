#include "taco/tensor.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <string>
#include <cstdint>
#include "benchmark.hpp"

using namespace taco;

int main(int argc, char **argv) {
  if(argc != 5){
    std::cerr << "wrong number of arguments" << std::endl;
  }

  std::string file_b = argv[1];
  std::string file_A1 = argv[2];
  std::string file_A2 = argv[3];
  std::string file_A3 = argv[4];

  Tensor<int64_t> A1 = read(file_A1, Format({Dense, Sparse}), true);
  Tensor<int64_t> A2 = read(file_A2, Format({Dense, Sparse}), true);
  Tensor<int64_t> A3 = read(file_A3, Format({Dense, Sparse}), true);

  Tensor<int64_t> b = read(file_b, Format({}), true);

  IndexVar i, j, k;

  b = A1(i, j) * A2(i, k) * A3(j, k);

  // Compile the expression
  b.compile();

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&b]() {
      b.setNeedsCompute(true);
    },
    [&b]() {
      b.compute();
    }
  );

  std::cout << time << std::endl;

  write(file_b, b);

  return 0;
}