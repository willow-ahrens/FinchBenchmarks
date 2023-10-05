#include "taco/tensor.h"
#include "taco.h"

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

Func roundFunc(){
  auto roundFunc = [](const std::vector<ir::Expr>& v) {
      return ir::BinOp::make(v[0], 0, "nearbyint(", " /** ", " **/ )"); 
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      return Region(v[0]);
  };
  Func roundfunc("round_func", roundFunc, algFunc);
  return roundfunc;
}

Func copyFunc(){
  auto copyFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], 0);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      return Union(l, Complement(l));
  };
  Func copy("copy_", copyFunc, algFunc);
  return copy;
}

int main(int argc, char **argv) {
    if(argc != 6){
        std::cerr << "wrong number of arguments" << std::endl;
    }

    // A = x*B + (1-x)*C
    std::string file_A = argv[1];
    std::string file_B = argv[2];
    std::string file_C = argv[3];
    std::string file_A_Dense = argv[5];
    double alpha       = std::stod(argv[4]);
    double beta        = 1 - alpha;

  TensorBase A_temp = read(file_A, Format({Dense, RLE}), true);
  A_temp.setName("A_temp");
  TensorBase B_temp = read(file_B, Format({Dense, RLE}), true);
  B_temp.setName("B_temp");
  TensorBase C_temp = read(file_C, Format({Dense, RLE}), true);
  C_temp.setName("C_temp");


  Tensor<uint8_t> A("A", A_temp.getDimensions(), A_temp.getFormat());
  Tensor<uint8_t> B("B", B_temp.getDimensions(), B_temp.getFormat());
  Tensor<uint8_t> C("C", C_temp.getDimensions(), C_temp.getFormat());

  IndexVar i("i"), j("j");

  A(i,j) = A_temp(i,j);
  A.evaluate();
  B(i,j) = B_temp(i,j);
  B.evaluate();
  C(i,j) = C_temp(i,j);
  C.evaluate();

  auto stmt = A(i, j) = roundFunc()(RLEPlus()(alpha*B(i, j), beta*C(i,j)));

  // Compile the expression
  A.setAssembleWhileCompute(true);
  A.compile();
  // A.printComputeIR(std::cout);

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

  Tensor<uint8_t> A_dense("A_dense", A.getDimensions(), {Dense, Dense});

  A_dense(i,j) = copyFunc()(A(i,j));
  A_dense.evaluate();
  write(file_A_Dense, A_dense);

  return 0;
}