#include <iostream>
#include "taco.h"

using namespace taco;

int main(int argc, char* argv[]) {
//  Format csr({Dense,Sparse});
//  Format csf({Sparse,Sparse,Sparse});
//  Format  sv({Sparse});
//
//  Tensor<double> A("A", {2,3},   csr, 1);
//  Tensor<double> B("B", {2,3,4}, csf, 2);
//  Tensor<double> c("c", {4},     sv,  3);
//
//  // Insert data into B and c
//  B(0,0,0) = 1.0;
//  B(1,2,0) = 2.0;
//  B(1,2,1) = 3.0;
//  c(0) = 4.0;
//  c(1) = 5.0;
//
//  IndexVar i("i"), j("j"), k("k");
//  A(i,j) = B(i,j,k) * c(k);
//
//  A.evaluate();
//  A.printComputeIR(std::cout);
//
//  std::cout << A << std::endl;

  Format  sv({RLE});
  Format  dv({Dense});
  Format  vb({VB});

  Tensor<double> A("A", {10},   sv, 10);
  Tensor<double> B("B", {10}, sv, 2);
  Tensor<double> C("C", {10},     dv,  3);

  // Insert data into B and c
  B(0) = 0.0;
  B(3) = 1.0;
  B(8) = 2.0;
  C(0) = 4.0;
  C(8) = 5.0;

  auto opFunc = [](const std::vector<ir::Expr>& v) {
    return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
    auto l = Region(v[0]);
    auto r = Region(v[1]);
    return Union(Union(l, r));
    return Union(Union(l, r), Union(Complement(l), Complement(r)));
  };
  std::initializer_list<Property> properties = {Identity(0), Commutative()};

  Func plus_("plus_", opFunc, algFunc);

  IndexVar i("i");
  A(i) = plus_(B(i), C(i));

//  A.setAssembleWhileCompute(true);
//  A.evaluate();
  A.compile();
  A.printComputeIR(std::cout);
//  A.printAssembleIR(std::cout);

//  std::cout << B << std::endl;
//  std::cout << C << std::endl;
//
//  std::cout << A << std::endl;
}
