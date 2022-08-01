// #include <iostream>
// #include <taco/lower/mode_format_vb.h>
// #include "../../src/lower/iteration_graph.h"
// #include "taco.h"
// #include "../../png/lodepng.h"

// #include <limits>
// #include <random>
// #include <variant>
// #include <algorithm>

// using namespace taco;

// const Format dv({Dense});
// const Format lz77f({LZ77});

// const IndexVar i("i");

// template <typename T>
// union GetBytes {
//     T value;
//     uint8_t bytes[sizeof(T)];
// };

// using Repeat = std::pair<uint16_t, uint16_t>;

// template <class T>
// using TempValue = std::variant<T,Repeat>;

// // helper type for the visitor #4
// template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// // explicit deduction guide (not needed as of C++20)
// template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// template <typename T>
// T get_value(const std::vector<uint8_t>& bytes, size_t pos){
//   T* ptr = (T*) &bytes[pos];
//   return *ptr;
// }

// template <typename T>
// void set_value(std::vector<uint8_t>& bytes, size_t pos, T val){
//   GetBytes<T> gb{val};
//   for (unsigned long i_=0; i_<sizeof(T); i_++){
//     bytes[pos+i_] = gb.bytes[i_];
//   }
// }

// template <typename T>
// void push_back(T arg, std::vector<uint8_t>& bytes, size_t& curr_count, bool& isValues, bool check = false){
//   GetBytes<T> gb;
//   gb.value = arg;

//   uint16_t mask = (uint16_t)0x7FFF;
//   uint16_t count = 0;
//   if (check) {
//     if (isValues && ((count = get_value<uint16_t>(bytes, curr_count)) < mask)) {
//       auto temp_curr_count = curr_count;
//       set_value<uint16_t>(bytes, curr_count, count + 1);
//       push_back<T>(arg, bytes, curr_count, isValues, false);
//       curr_count = temp_curr_count;
//     } else {
//       push_back<uint16_t>(1, bytes, curr_count, isValues, false);
//       auto temp_curr_count = size_t(bytes.empty() ? 0 : bytes.size()-2);
//       push_back<T>(arg, bytes, curr_count, isValues, false);
//       curr_count = temp_curr_count;
//     }
//     isValues = true;
//   } else {
//     for (unsigned long i_=0; i_<sizeof(T); i_++){
//       bytes.push_back(gb.bytes[i_]);
//     }
//     isValues = false;
//     curr_count = 0;
//   }
// }

// template <typename T>
// std::vector<uint8_t> packLZ77_bytes(std::vector<TempValue<T>> vals){
//   std::vector<uint8_t> bytes;
//   size_t curr_count = 0;
//   bool isValues = false;
//   const auto runMask = (uint16_t)~0x7FFF;
//   for (auto& val : vals){
//     std::visit(overloaded {
//             [&](T arg) { push_back(arg, bytes, curr_count, isValues, true); },
//             [&](std::pair<uint16_t, uint16_t> arg) {
//                 push_back<uint16_t>(arg.second | runMask, bytes, curr_count, isValues);
//                 push_back<uint16_t>(arg.first, bytes, curr_count, isValues);
//             }
//     }, val);
//   }
//   return bytes;
// }


// template <typename T>
// std::pair<std::vector<uint8_t>, int> packLZ77(std::vector<TempValue<T>> vals){
//   std::vector<uint8_t> bytes = packLZ77_bytes(vals);
//   return {bytes,bytes.size()};
// //  int size = bytes.size();
// //  while(bytes.size() % sizeof(T) != 0){
// //    bytes.push_back(0);
// //  }
// //  T* bytes_data = (T*) bytes.data();
// //  std::vector<T> values(bytes_data, bytes_data + (bytes.size() / sizeof(T)));
// //
// //  return {values, size};
// }

// Index makeLZ77Index(const std::vector<int>& rowptr) {
//   return Index(lz77f,
//                {ModeIndex({makeArray(rowptr)})});
// }

// /// Factory function to construct a compressed sparse row (CSR) matrix.
// template<typename T>
// TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
//                     const std::vector<int>& pos, const std::vector<uint8_t>& vals) {
//   taco_uassert(dimensions.size() == 1);
//   Tensor<T> tensor(name, dimensions, {LZ77});
//   auto storage = tensor.getStorage();
//   storage.setIndex(makeLZ77Index(pos));
//   storage.setValues(makeArray(vals));
//   tensor.setStorage(storage);
//   return std::move(tensor);
// }

// Tensor<double> lz77_zeros(std::string name) {
//   auto packed = packLZ77<double>({0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0});
//   return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
// }

// Tensor<double> lz77_one_rle(std::string name, double val) {
//   auto packed = packLZ77<double>({val,Repeat{1,9},0.0});
//   return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
// }


// Tensor<double> lz77_two_repeat(std::string name, double val1, double val2) {
//   auto packed = packLZ77<double>({val1,val2,Repeat{2,8},0.0});
//   return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
// }

// Tensor<double> lz77_repeat_twice(std::string name, double val1, double val2) {
//   auto packed = packLZ77<double>({val1,Repeat{1,3},val1,val2,Repeat{2,4},0.0});
//   return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
// }

// Tensor<double> lz77_three_repeat(std::string name, double val1, double val2, double val3) {
//   auto packed = packLZ77<double>({val1,val2,val3,Repeat{3,7},0.0});
//   return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
// }

// Tensor<double> lz77_1(std::string name) {
//   auto packed = packLZ77<double>({0.0,1.0,2.0,Repeat{1,3},3.0,4.0,5.0,Repeat{3,1},0.0});
//   return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
// }

// Tensor<double> lz77_2(std::string name) {
//   auto packed = packLZ77<double>({0.0,1.0,2.0,Repeat{1,2},3.0,4.0,5.0,Repeat{3,2},0.0});
//   return makeLZ77<double>(name, {11}, {0, packed.second}, packed.first);
// }

// std::default_random_engine gen(0);

// Func getCopyFunc(){
//   auto copyFunc = [](const std::vector<ir::Expr>& v) {
//       return ir::Add::make(v[0], 0);
//   };
//   auto algFunc = [](const std::vector<IndexExpr>& v) {
//       auto l = Region(v[0]);
//       return Union(l, Complement(l));
//   };
//   Func copy("copy_", copyFunc, algFunc);
//   return copy;
// }

// Func getPlusFunc(){
//   auto plusFunc = [](const std::vector<ir::Expr>& v) {
//       return ir::Add::make(v[0], v[1]);
//   };
//   auto algFunc = [](const std::vector<IndexExpr>& v) {
//       auto l = Region(v[0]);
//       auto r = Region(v[1]);
//       return Union(Union(l, r), Union(Complement(l), Complement(r)));
//   };
//   Func plus_("plus_", plusFunc, algFunc);
//   return plus_;
// }

// Func getTimesFunc(){
//   auto plusFunc = [](const std::vector<ir::Expr>& v) {
//       return ir::Mul::make(v[0], v[1]);
//   };
//   auto algFunc = [](const std::vector<IndexExpr>& v) {
//       auto l = Region(v[0]);
//       auto r = Region(v[1]);
//       return Union(Union(l, r), Union(Complement(l), Complement(r)));
//   };
//   Func plus_("times", plusFunc, algFunc);
//   return plus_;
// }


// int test_zeros() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, lz77f, 0);
//   Tensor<double> B = lz77_zeros("B");
//   Tensor<double> C = lz77_zeros("C");
//   Tensor<double> result("result", {11}, dv, 0);

//   A(i) = plus_(B(i), C(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.printComputeIR(std::cout);
//   A.compute();

//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);
//   Tensor<double> C_("dC", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
//   return 0;
// }

// int test_one_rle() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, lz77f, 0);
//   Tensor<double> B = lz77_one_rle("B", 5);
//   Tensor<double> C = lz77_one_rle("C", 7);
//   Tensor<double> result("result", {11}, dv, 0);

//   A(i) = plus_(B(i), C(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.printComputeIR(std::cout);
//   A.compute();

//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);
//   Tensor<double> C_("dC", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
//   return 0;
// }

// int test_repeat_two() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, lz77f, 0);
//   Tensor<double> B = lz77_two_repeat("B", 1,2);
//   Tensor<double> C = lz77_two_repeat("C", 3,4);
//   Tensor<double> result("result", {11}, dv, 0);

//   A(i) = plus_(B(i), C(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.compute();

//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);
//   Tensor<double> C_("dC", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
//   return 0;
// }

// int test_mixed_two_three() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, lz77f, 0);
//   Tensor<double> B = lz77_two_repeat("B", 1,2);
//   Tensor<double> C = lz77_three_repeat("C", 3,4,5);
//   Tensor<double> result("result", {11}, dv, 0);

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   A(i) = plus_(B(i), C(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.printComputeIR(std::cout);
//   A.compute();

//   std::cout << A << std::endl;

//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);
//   Tensor<double> C_("dC", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
//   return 0;
// }

// int test_mixed() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, lz77f, 0);
//   Tensor<double> B = lz77_repeat_twice("B", 1,2);
//   Tensor<double> C = lz77_three_repeat("C", 3,4,5);
//   Tensor<double> result("result", {11}, dv, 0);

//   A(i) = plus_(B(i), C(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.compute();

//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);
//   Tensor<double> C_("dC", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
//   return 0;
// }

// void test_repeat_two_csr() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, lz77f, 0);
// //  Tensor<double> A("A", {11}, dv, 0);
//   Tensor<double> B = lz77_two_repeat("B", 1,2);
//   Tensor<double> C("C", {11}, {Compressed}, 0);
//   Tensor<double> result("result", {11}, dv, 0);

//   C(0) = 1;
//   C(5) = 2;
//   C(9) = 5;
//   C(10) = 0;

//   A(i) = plus_(B(i), C(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.printComputeIR(std::cout);
//   A.compute();

//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);
//   Tensor<double> C_("dC", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
// }

// Index makeLZ77RGBAIndex(const std::vector<int>& rowptr) {
//   return Index({LZ77, Dense},
//                {ModeIndex({makeArray(rowptr)}),
//                 ModeIndex({makeArray({4})})});
// }

// /// Factory function to construct a compressed sparse row (CSR) matrix.
// template<typename T>
// TensorBase makeLZ77RGBA(const std::string& name, const std::vector<int>& dimensions,
//                     const std::vector<int>& pos, const std::vector<T>& vals) {
//   taco_uassert(dimensions.size() == 2);
//   taco_uassert(dimensions[1] == 4);
//   Tensor<T> tensor(name, dimensions, {LZ77, Dense});
//   auto storage = tensor.getStorage();
//   storage.setIndex(makeLZ77RGBAIndex(pos));
//   storage.setValues(makeArray(vals));
//   tensor.setStorage(storage);
//   return std::move(tensor);
// }

// struct RGBA {
//     uint8_t r;
//     uint8_t g;
//     uint8_t b;
//     uint8_t a;
// };

// Tensor<uint8_t> lz77_dense_RGBA(std::string name, RGBA c) {
//   auto packed = packLZ77_bytes<RGBA>({c,Repeat{2,8},c});
//   return makeLZ77RGBA<uint8_t>(name, {11,4}, {0, (int)packed.size()}, packed);
// }

// void test_lz77_dense() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<uint8_t> A("A", {11,4}, {Dense, Dense}, 0);
//   Tensor<uint8_t> B = lz77_dense_RGBA("B", {1,2,3,4});
//   const IndexVar j("j");

//   A(i,j) = copy(B(i,j));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.printComputeIR(std::cout);
//   A.compute();


//   std::cout << B << std::endl;
//   std::cout << A << std::endl;
// }

// void test_values(){
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, dv, 0);
//   Tensor<double> B = lz77_two_repeat("B", 1,2);

//   A(i) = copy(B(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.compute();

//   Tensor<double> result("result", {11}, dv, 0);
//   result(0) = 1;
//   result(1) = 2;
//   result(2) = 1;
//   result(3) = 2;
//   result(4) = 1;
//   result(5) = 2;
//   result(6) = 1;
//   result(7) = 2;
//   result(8) = 1;
//   result(9) = 2;
//   result(10) = 0;
//   result.pack();

//   std::cout << B << std::endl;
//   std::cout << A << std::endl;
//   std::cout << result << std::endl;
// }

// Func getPlusRleFunc(){
//   auto plusFunc = [](const std::vector<ir::Expr>& v) {
//       return ir::Add::make(v[0], v[1]);
//   };
//   auto algFunc = [](const std::vector<IndexExpr>& v) {
//       auto l = Region(v[0]);
//       auto r = Region(v[1]);
//       return Union(l, r);
//   };
//   Func plus_("plus_", plusFunc, algFunc);
//   return plus_;
// }


// Func getTimesRleFunc(){
//   auto plusFunc = [](const std::vector<ir::Expr>& v) {
//       return ir::Mul::make(v[0], v[1]);
//   };
//   auto algFunc = [](const std::vector<IndexExpr>& v) {
//       auto l = Region(v[0]);
//       auto r = Region(v[1]);
//       return Union(l, r);
//   };
//   Func times("times", plusFunc, algFunc);
//   return times;
// }


// void test_rle_values(){
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();

//   Tensor<double> A("A", {11}, dv, 0);
//   Tensor<double> B("B", {11}, {RLE}, 0);

//   B(0) = 3;
//   B(8) = 7;

//   A(i) = copy(B(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.compute();

//   Tensor<double> result("result", {11}, dv, 0);
//   result(0) = 3;
//   result(1) = 3;
//   result(2) = 3;
//   result(3) = 3;
//   result(4) = 3;
//   result(5) = 3;
//   result(6) = 3;
//   result(7) = 3;
//   result(8) = 7;
//   result(9) = 7;
//   result(10) = 7;
//   result.pack();

//   std::cout << B << std::endl;
//   std::cout << A << std::endl;
//   std::cout << result << std::endl;
//   A.printComputeIR(std::cout);
// }

// void test_rle() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusRleFunc();

//   Tensor<double> A("A", {100}, {RLE}, 0);
//   Tensor<double> B("B", {100}, {RLE}, 0);
//   Tensor<double> C("C", {100}, {RLE}, 0);

//   B(0) = 1;
//   B(4) = 3;

//   C(0) = 10;
//   C(5) = 20;
//   C(9) = 50;
//   C(10) = 0;

//   A(i) = plus_(B(i), C(i));
// //  A.setAssembleWhileCompute(true);
//   A.compile();
//   A.assemble();
// //  A.printComputeIR(std::cout);
//   A.compute();

//   Tensor<double> result("result", {100}, dv, 0);
//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.printComputeIR(std::cout);
//   result.compute();

//   Tensor<double> A_("dA", {100},   dv, 0);
//   Tensor<double> B_("dB", {100},   dv, 0);
//   Tensor<double> C_("dC", {100},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
// }

// void test_rle_rgba(){
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusRleFunc();

//   Tensor<double> A("A", {5,4}, {RLE_size(4),Dense}, 0);
//   Tensor<double> B("B", {5,4}, {RLE_size(4),Dense}, 0);
//   Tensor<double> C("C", {5,4}, {RLE_size(4),Dense}, 0);
//   Tensor<double> result("result", {5,4}, {Dense, Dense}, 0);

//   C(0,0) = 1;
//   C(0,1) = 2;
//   C(0,2) = 3;
//   C(0,3) = 4;

//   B(0,0) = 5;
//   B(0,1) = 6;
//   B(0,2) = 7;
//   B(0,3) = 8;

//   B(3,0) = 100;
//   B(3,1) = 100;
//   B(3,2) = 100;
//   B(3,3) = 100;

//   IndexVar i("i"), j("j"), c("c");
//   A(i,c) = plus_(B(i,c), C(i,c));
// //  A.setAssembleWhileCompute(true);
//   A.compile();
//   A.assemble();
//   A.printComputeIR(std::cout);
//   A.compute();

//   result(i,c) = copy(A(i,c));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {5,4}, {Dense, Dense}, 0);
//   Tensor<double> B_("dB", {5,4}, {Dense, Dense}, 0);
//   Tensor<double> C_("dC", {5,4}, {Dense, Dense}, 0);

//   B_(i,c) = copy(B(i,c));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i,c) = copy(C(i,c));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i,c) = B_(i,c) + C_(i,c);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
// }

// ir::Expr ternaryOp(const ir::Expr& c, const ir::Expr& a, const ir::Expr& b){
//   // c ? a : b
//   ir::Expr a_b = ir::BinOp::make(a,b, " : ");
//   return ir::BinOp::make(c, a_b, "(", " ? ", ")");
// }

// Func getBrightenFunc(uint8_t brightness, bool full){
//   auto brighten = [=](const std::vector<ir::Expr>& v) {
//       auto sum = ir::Add::make(v[0], brightness);
//       return ternaryOp(ir::Gt::make(sum, 255), 255, sum);
//   };
//   auto algFunc = [=](const std::vector<IndexExpr>& v) {
//       auto l = Region(v[0]);
//       if (full){
//         return IterationAlgebra(Union(l, Complement(l)));
//       } else {
//         return IterationAlgebra(l);
//       }
//   };
//   Func plus_("plus_", brighten, algFunc);
//   return plus_;
// }

// void test_brighten() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();
//   Func brighten = getBrightenFunc(20, true);

//   Tensor<double> A("A", {11}, lz77f, 0);
//   Tensor<double> B = lz77_three_repeat("B", 1,2,250);

//   std::cout << B << std::endl;

//   A(i) = brighten(B(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.printComputeIR(std::cout);
//   A.compute();

//   std::cout << A << std::endl;

//   Tensor<double> result("result", {11}, dv, 0);
//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   std::cout << result << std::endl;

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

// //  A_(i) = brighten(B_(i));
// //  A_.setAssembleWhileCompute(true);
// //  A_.compile();
// //  A_.compute();

//   std::cout << B << std::endl;

//   std::cout << B_ << std::endl;

//   std::cout << A << std::endl;
// //  std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
// }

// std::vector<uint8_t> encode_lz77(const std::vector<uint8_t> in);
// std::vector<uint8_t> unpackLZ77_bytes(std::vector<uint8_t> bytes);
// void test_compress(){
//   auto out = encode_lz77({1,1,1,1,1,1,4,5,6,4,5,6,4,5,6,1,0,1,5,128});
// //  auto out = encode_lz77({0,1,2,3,4,5,6,7,8,9,10});
//   for (auto& o : out){
//     std::cout << (unsigned) o << " ";
//   }
//   std::cout << std::endl;
//   auto inflated = unpackLZ77_bytes(out);
//   for (auto& o : inflated){
//     std::cout << (unsigned) o << " ";
//   }
//   std::cout << std::endl;

// }

// int test_three_ops() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusFunc();
//   Func times = getTimesFunc();

//   Tensor<double> A("A", {11}, lz77f, 0);
//   Tensor<double> B = lz77_repeat_twice("B", 1,2);
//   Tensor<double> C = lz77_three_repeat("C", 3,4,5);
//   Tensor<double> D = lz77_one_rle("D", 6);
//   Tensor<double> result("result", {11}, dv, 0);

//   A(i) = plus_(B(i), times(C(i), D(i)));
//   A.setAssembleWhileCompute(true);
//   A.compile();
//   A.compute();
//   A.printComputeIR(std::cout);

//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   Tensor<double> A_("dA", {11},   dv, 0);
//   Tensor<double> B_("dB", {11},   dv, 0);
//   Tensor<double> C_("dC", {11},   dv, 0);
//   Tensor<double> D_("dD", {11},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   D_(i) = copy(D(i));
//   D_.setAssembleWhileCompute(true);
//   D_.compile();
//   D_.compute();

//   A_(i) = B_(i) + (C_(i) * D_(i));

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;
//   std::cout << D << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;
//   std::cout << D_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
//   return 0;
// }

// Func getPlusUniverseFunc(){
//   auto plusFunc = [](const std::vector<ir::Expr>& v) {
//       return ir::Add::make(v[0], v[1]);
//   };
//   auto algFunc = [](const std::vector<IndexExpr>& v) {
//       auto l = Region(v[0]);
//       auto r = Region(v[1]);
//       return Union(Union(l, r), Union(Complement(l), Complement(r)));
//   };
//   Func plus_("plus_", plusFunc, algFunc);
//   return plus_;
// }

// void test_rle_csr() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusUniverseFunc();

//   Tensor<double> A("A", {10}, {RLE}, 0);
//   Tensor<double> B("B", {10}, {RLE}, 0);
//   Tensor<double> C("C", {10}, {Sparse}, 0);

//   B(0) = 1;
//   B(4) = 3;

//   C(0) = 10;
//   C(5) = 20;
//   C(9) = 50;

//   A(i) = plus_(B(i), C(i));
//   A.setAssembleWhileCompute(true);
//   A.compile();
// //  A.assemble();
//   A.printComputeIR(std::cout);
//   A.compute();

//   Tensor<double> result("result", {10}, dv, 0);
//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
// //  result.printComputeIR(std::cout);
//   result.compute();

//   Tensor<double> A_("dA", {10},   dv, 0);
//   Tensor<double> B_("dB", {10},   dv, 0);
//   Tensor<double> C_("dC", {10},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
// }

// void test_rle_rle() {
//   Func copy = getCopyFunc();
//   Func plus_ = getPlusRleFunc();

//   Tensor<double> A("A", {10}, {RLE}, 0);
//   Tensor<double> B("B", {10}, {RLE}, 0);
//   Tensor<double> C("C", {10}, {RLE}, 0);

//   B(0) = 1;
//   B(4) = 3;

//   C(0) = 10;
//   C(5) = 20;
//   C(9) = 50;
//   C(10) = 0;

//   A(i) = plus_(B(i), C(i));
// //  A.setAssembleWhileCompute(true);
//   A.compile();
//   A.assemble();
//   A.printComputeIR(std::cout);
//   A.compute();

//   Tensor<double> result("result", {10}, dv, 0);
//   result(i) = copy(A(i));
//   result.setAssembleWhileCompute(true);
//   result.compile();
// //  result.printComputeIR(std::cout);
//   result.compute();

//   Tensor<double> A_("dA", {10},   dv, 0);
//   Tensor<double> B_("dB", {10},   dv, 0);
//   Tensor<double> C_("dC", {10},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   A_(i) = B_(i) + C_(i);

//   A_.setAssembleWhileCompute(true);
//   A_.compile();
//   A_.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << A_ << std::endl;
//   std::cout << result << std::endl;
// }


// void test_rle_reduction() {
//   Func copy = getCopyFunc();
//   Func times = getTimesRleFunc();

//   Tensor<double> A("A");
//   Tensor<double> B("B", {10}, {RLE}, 0);
//   Tensor<double> C("C", {10}, {RLE}, 0);

//   B(0) = 1;
//   B(4) = 3;

//   C(0) = 10;
//   C(5) = 20;
//   C(9) = 50;
//   C(10) = 0;

//   A = times(B(i), C(i));
// //  A.setAssembleWhileCompute(true);
//   A.compile();
//   A.assemble();
//   A.printComputeIR(std::cout);
//   A.compute();

//   Tensor<double> result("result");
//   Tensor<double> B_("dB", {10},   dv, 0);
//   Tensor<double> C_("dC", {10},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   result = B_(i) * C_(i);

//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << result << std::endl;
// }

// void test_rle_dense_reduction() {
//   Func copy = getCopyFunc();
//   Func times = getTimesRleFunc();

//   Tensor<double> A("A");
//   Tensor<double> B("B", {10}, {RLE}, 0);
//   Tensor<double> C("C", {10}, {Dense}, 0);

//   B(0) = 1;
//   B(4) = 3;

//   C(0) = 10;
//   C(1) = 10;
//   C(2) = 10;
//   C(3) = 10;
//   C(4) = 10;
//   C(5) = 20;
//   C(6) = 20;
//   C(7) = 20;
//   C(8) = 20;
//   C(9) = 50;
//   C(10) = 0;

//   A = times(B(i), C(i));
// //  A.setAssembleWhileCompute(true);
//   A.compile();
//   A.assemble();
//   A.printComputeIR(std::cout);
//   A.compute();

//   Tensor<double> result("result");
//   Tensor<double> B_("dB", {10},   dv, 0);
//   Tensor<double> C_("dC", {10},   dv, 0);

//   B_(i) = copy(B(i));
//   B_.setAssembleWhileCompute(true);
//   B_.compile();
//   B_.compute();

//   C_(i) = copy(C(i));
//   C_.setAssembleWhileCompute(true);
//   C_.compile();
//   C_.compute();

//   result = B_(i) * C_(i);

//   result.setAssembleWhileCompute(true);
//   result.compile();
//   result.compute();

//   std::cout << B << std::endl;
//   std::cout << C << std::endl << std::endl;

//   std::cout << B_ << std::endl;
//   std::cout << C_ << std::endl << std::endl;

//   std::cout << A << std::endl;
//   std::cout << result << std::endl;
// }

int main() {
//  test_zeros();
//  test_one_rle();
//  test_repeat_two();
////////  test_mixed_two_three();
//  test_mixed();
//  test_repeat_two_csr();
//  test_lz77_dense();
//  test_values();
//  test_rle_values();

//  test_rle();
//  test_rle_rgba();

//  test_brighten();
//  test_compress();

//  test_three_ops();
  // test_rle_csr();

//  test_rle_rle();

//  test_rle_reduction();
//  test_rle_dense_reduction();
  return 0;
}