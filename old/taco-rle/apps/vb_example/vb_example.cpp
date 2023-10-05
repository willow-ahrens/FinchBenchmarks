#include <iostream>
#include <taco/lower/mode_format_vb.h>
#include <dlfcn.h>
#include "../../src/lower/iteration_graph.h"
#include "taco.h"

using namespace taco;
using namespace std;

Index makeVBVectorIndex(const std::vector<int>& rowptr, const std::vector<int>& colidx) {
  return Index({Dense, VB},
               {ModeIndex({makeArray({(int)rowptr.size()})}),
                ModeIndex({makeArray(rowptr), makeArray(colidx)})});
}


/// Factory function to construct a compressed sparse row (CSR) matrix.
template<typename T>
TensorBase makeVBVector(const std::string& name, const std::vector<int>& dimensions,
                   const std::vector<int>& pos,
                   const std::vector<int>& crd,
                   const std::vector<T>& vals) {
  taco_uassert(dimensions.size() == 2) << error::requires_matrix;
  Tensor<T> tensor(name, dimensions, {Dense, VB});
  auto storage = tensor.getStorage();
  storage.setIndex(makeVBVectorIndex(pos, crd));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

Tensor<int> vb_1(std::string name) {
  return makeVBVector<int>(name, {3,6},
                         {0, 3, 5, 6},
                         {0, 3, 5},
                         {0,1,2,  3,4, 5});
}

Tensor<int> vb_2(std::string name) {
  return makeVBVector<int>(name, {1,6},
                           {0, 3},
                           {3},
                           {4,5,6});
}

Tensor<int> vb_1_sparse(std::string name) {
  return makeVBVector<int>(name, {3,6},
                           {0, 3, 5, 6},
                           {0, 3, 5},
                           {0,1,2,  3,4, 5});
}

Tensor<int> vb_2_sparse(std::string name) {
  return makeVBVector<int>(name, {1,6},
                           {0, 3},
                           {3},
                           {4,5,6});
}

inline
vector<void*> packArguments(const vector<TensorStorage>& args) {
  vector<void*> arguments;
  arguments.reserve(args.size());
  for (auto& arg : args) {
    arguments.push_back(static_cast<taco_tensor_t*>(arg));
  }
  return arguments;
}

inline
void unpackResults(size_t numResults, const vector<void*> arguments,
                   const vector<TensorStorage>& args) {
  for (size_t i = 0; i < numResults; i++) {
    taco_tensor_t* tensorData = ((taco_tensor_t*)arguments[i]);
    TensorStorage storage = args[i];
    Format format = storage.getFormat();

    vector<ModeIndex> modeIndices;
    size_t num = 1;
    for (int i = 0; i < storage.getOrder(); i++) {
      ModeFormat modeType = format.getModeFormats()[i];
      if (modeType.getName() == Dense.getName()) {
        Array size = makeArray({*(int*)tensorData->indices[i][0]});
        modeIndices.push_back(ModeIndex({size}));
        num *= ((int*)tensorData->indices[i][0])[0];
      } else if (modeType.getName() == Sparse.getName()) {
        auto size = ((int*)tensorData->indices[i][0])[num];
        Array pos = Array(type<int>(), tensorData->indices[i][0],
                          num+1, Array::UserOwns);
        Array idx = Array(type<int>(), tensorData->indices[i][1],
                          size, Array::UserOwns);
        modeIndices.push_back(ModeIndex({pos, idx}));
        num = size;
      } else {
        taco_not_supported_yet;
      }
    }
    storage.setIndex(Index(format, modeIndices));
    storage.setValues(Array(storage.getComponentType(), tensorData->vals, num));
  }
}

IndexExpr plusT(IndexExpr lhs, IndexExpr rhs){
  auto plusFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      auto r = Region(v[1]);
      return Union(Union(l, r), Union(Complement(l), Complement(r)));
  };
  Func plus_("plus_", plusFunc, algFunc);
  return plus_(lhs, rhs);
}

int main(int argc, char* argv[]) {
//  Format  sv({RLE});
  Format  dv({Dense});
//  Format  vb({VB});

  Tensor<int> A("A", {3,6}, {Dense, Dense}, 0);
  Tensor<int> C("C", {6}, {Dense}, 0);
  Tensor<int> B = vb_1("B");
  Tensor<int> D = vb_2("D");
//  Tensor<double> B("B", {10}, sv, 2);
//  Tensor<double> C("C", {15},     sv,  0);

//
//  // Insert data into B and c
//  C(0) = 4.0;
//  C(8) = 5.0;

  std::cout << B << std::endl;
  std::cout << D << std::endl;

  IndexVar i("i"), B_i_blk("IB"), D_i_blk("ID");
//  A(B_i_blk, i) = B(B_i_blk, i) + D(B_i_blk, i);

//  {
//    IterationGraph iterationGraph = IterationGraph::make(A.getAssignment());
//    iterationGraph.printAsDot(std::cout);
//    std::cout << iterationGraph << std::endl;
//  }

//  A.compile();
//  A.printComputeIR(std::cout);
//  A.assemble();
//  A.compute();

//  auto opFunc = [](const std::vector<ir::Expr>& v) {
//    return ir::Add::make(v[0], v[1]);
//  };
//  auto algFunc = [](const std::vector<IndexExpr>& v) {
//    auto l = Region(v[0]);
//    auto r = Region(v[1]);
//    return Union(Union(l, r));
////    return Union(Union(l, r), Union(Complement(l), Complement(r)));
//  };
//  std::initializer_list<Property> properties = {Identity(0), Commutative()};
//
//  Func plus_("plus_", opFunc, algFunc);
//
//  IndexVar i("i"), i_blk("i_blk");
//  A(i) = plus_(B(i_blk, i), C(i));
//
//  A.setAssembleWhileCompute(true);
////  A.evaluate();
//  A.compile();
//  A.printComputeIR(std::cout);
////  A.assemble();
////  A.compute();
//
////  A.printAssembleIR(std::cout);
//
//
////  std::cout << A << std::endl;

  IndexVar k("k"), l("l");

  // where(where(C(i) = tIB + tID, forall(ID, tID += D(ID,i))), forall(IB, tIB += B(IB,i)))
//  auto res = forall(B_i_blk, forall(D_i_blk, forall(i, C(i) += B(B_i_blk, i) + D(D_i_blk, i))));
  Tensor<int> tIB("tIB", {6}, {Dense}, 0);
//  tIB(i) = sum(k, B(k, i));
  Tensor<int> tID("tID", {6}, {Dense}, 0);
//  tID(i) = sum(l, D(l, i));

//  auto res = where(where(forall(i, C(i) = tIB(i) + tID(i)), forall(i, forall(l, tID(i) += D(l,i)))), forall(i, forall(k, tIB(i) += B(k,i))));
  auto res = where(where(forall(i, C(i) = tIB(i) + tID(i)),
                          forall(k, forall(i, tIB(i) += B(k, i)))),
                          forall(l, forall(i, tID(i) += D(l, i))));
//  res.
//  C(i) = tIB(i) + tID(i);
  C(i) = plusT(sum(k, B(k, i)), sum(l, D(l, i)));
//  C(i) = sum(k, B(k, i));
  C.compile(res);
  C.assemble();
  C.printComputeIR(std::cout);


//  std::shared_ptr<ir::Module> module(new ir::Module);
//  auto lib_handle = dlopen(std::string("/Users/danieldonenfeld/Developer/taco/apps/vb_example/vb_handwritten.so").data(), RTLD_NOW | RTLD_LOCAL);
//  taco_uassert(lib_handle) << "Failed to load generated code, error is: " << dlerror();
//
//  module->lib_handle = lib_handle;
//
//  void* evaluate = dlsym(lib_handle, std::string("compute").data());
//  void* compute  = dlsym(lib_handle, std::string("compute").data());
//  auto kernel = Kernel(res, module, evaluate, nullptr, compute);
//
//  std::vector<TensorStorage> st = {C.getStorage(), B.getStorage(), D.getStorage()};
//  auto packed = packArguments(st);
//  int result = module->callFuncPacked("compute", packed.data());
//  unpackResults(1, packed, st);
//
//  C.setNeedsCompute(false);
//
//  for (int q=0; q<6;q++) {
//    std::cout  << ((int*)((taco_tensor_t *) packed[0])->vals)[q] << ", ";
//  }
//  std::cout << std::endl;

  std::cout << C << std::endl;

  return 0;
}
