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

  std::string file_c = argv[1];
  std::string file_A = argv[2];
  std::string file_A2 = argv[3];
  std::string file_AT = argv[4];

  Tensor<int64_t> A = read(file_A, Format({Dense, Sparse}), true);
  Tensor<int64_t> A2 = read(file_A2, Format({Dense, Sparse}), true);
  Tensor<int64_t> AT = read(file_AT, Format({Dense, Sparse}), true);

  Tensor<int64_t> c = read(file_c, Format({}), true);

  IndexVar i, j, k;

  c = A(i, j) * A2(j, k) * AT(i, k);

  // Compile the expression
  c.compile();

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&c]() {
      c.setNeedsCompute(true);
    },
    [&c]() {
      c.compute();
    }
  );

  std::cout << time << std::endl;

  write(file_c, c);

  return 0;
}