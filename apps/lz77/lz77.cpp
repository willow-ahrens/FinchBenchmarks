//#include <iostream>
//#include <taco/lower/mode_format_vb.h>
//#include "../../src/lower/iteration_graph.h"
//#include "taco.h"
//
//using namespace taco;
//
//Index makeLZ77Index(const std::vector<int>& rowptr, const std::vector<int>& dist,
//                        const std::vector<int>& runs) {
//  return Index({LZ77},
//               {ModeIndex({makeArray(rowptr), makeArray(dist), makeArray(runs)})});
//}
//
//
///// Factory function to construct a compressed sparse row (CSR) matrix.
//template<typename T>
//TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
//                   const std::vector<int>& pos,
//                   const std::vector<int>& dist,
//                   const std::vector<int>& runs,
//                   const std::vector<T>& vals) {
//  taco_uassert(dimensions.size() == 1);
//  Tensor<T> tensor(name, dimensions, {LZ77});
//  auto storage = tensor.getStorage();
//  storage.setIndex(makeLZ77Index(pos, dist, runs));
//  storage.setValues(makeArray(vals));
//  tensor.setStorage(storage);
//  return std::move(tensor);
//}
//
//Tensor<double> lz77_1(std::string name) {
//  return makeLZ77<double>(name, {11},
//                         {0, 4},
//                         {0,  2,  0},
//                         {0,  8,  0},
//                         {1.0,3.0,0.0});
//}
//
//Tensor<double> lz77_2(std::string name) {
//  return makeLZ77<double>(name, {11},
//                          {0, 3},
//                          {1,  0},
//                          {9,  0},
//                          {2.0,0.0});
//}
//
//int main(int argc, char* argv[]) {
//  Format  dv({Dense});
//  Format  LZ77f({LZ77});
//  IndexVar i("i");
//
//
//  Tensor<double> A("A", {11},   LZ77f, 0);
//  Tensor<double> B = lz77_1("B");
//  Tensor<double> C = lz77_2("C");
//  Tensor<double> D("D", {11},   dv, 0);
//  Tensor<double> B_("dB", {11},   dv, 0);
//  Tensor<double> C_("dC", {11},   dv, 0);
//
//  B_(i) = B(i) + D(i);
//  C_(i) = C(i) + D(i);
//
//  auto opFunc = [](const std::vector<ir::Expr>& v) {
//    return ir::Add::make(v[0], v[1]);
//  };
//  auto algFunc = [](const std::vector<IndexExpr>& v) {
//    auto l = Region(v[0]);
//    auto r = Region(v[1]);
////    return Union(Union(l, r));
//    return Union(Union(l, r), Union(Complement(l), Complement(r)));
//  };
//  Func plus_("plus_", opFunc, algFunc);
//
//  std::cout << B << std::endl;
//  std::cout << B_ << std::endl;
//
//  std::cout << C << std::endl;
//  std::cout << C_ << std::endl;
//
//  A(i) = plus_(B(i), C(i));
//
//  A.setAssembleWhileCompute(true);
//  A.compile();
//  A.printComputeIR(std::cout);
////  A.assemble();
//  A.compute();
//
//  std::cout << A << std::endl;
//
//  D(i) = plus_(B(i), C(i));
//  D.setAssembleWhileCompute(true);
//  D.compile();
//  D.compute();
//
//  std::cout << D << std::endl;
//
//}
