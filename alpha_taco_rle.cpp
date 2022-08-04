#include "taco/tensor.h"
#include "taco.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <iostream>
#include <string>
#include "benchmark.hpp"

using namespace taco;

Func RLEPlus(){
  auto plusFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      auto r = Region(v[1]);
      return Union(l, r);
  };
  Func plus_("plus_", plusFunc, algFunc);
  return plus_;
}

int main(int argc, char **argv) {
    if(argc != 5){
        std::cerr << "wrong number of arguments" << std::endl;
    }

    // A = x*B + (1-x)*C
    std::string file_A = argv[1];
    std::string file_B = argv[2];
    std::string file_C = argv[3];
    double alpha       = std::stod(argv[4]);
    double beta        = 1 - alpha;

  TensorBase A = read(file_A, Format({Dense, RLE}), true);
  TensorBase B = read(file_B, Format({Dense, RLE}), true);
  TensorBase C = read(file_C, Format({Dense, RLE}), true);

  IndexVar i, j;

  auto stmt = A(i, j) = RLEPlus()(alpha*B(i, j), beta*C(i,j));

  // Compile the expression
  A.setAssembleWhileCompute(true);
  A.compile();

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&A]() {
    //   y.setNeedsAssemble(true);
      A.setNeedsCompute(true);
    },
    [&A]() {
    //   y.assemble();
      A.compute();
    }
  );

  std::cout << time << std::endl;

  write(file_A, A);

  return 0;
}