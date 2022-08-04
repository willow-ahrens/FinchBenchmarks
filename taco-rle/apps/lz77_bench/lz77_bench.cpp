#include "../test/test.h"
#include "taco/tensor.h"
#include <benchmark/benchmark.h>
#include "../../src/lower/iteration_graph.h"
#include "taco.h"

#include <random>
#include <variant>
#include <climits>
#include <limits>

extern "C" {
  #include "lz_sum_kernel.h"
}

using namespace taco;

#ifndef SEED
#define SEED 0
#endif
std::default_random_engine gen(SEED);

#ifndef TENSOR_TYPE
#define TENSOR_TYPE uint8_t
#endif

const Format dv({Dense});
const Format lz77f({LZ77});

template <typename T>
union GetBytes {
    T value;
    uint8_t bytes[sizeof(T)];
};

using Repeat = std::pair<uint16_t, uint16_t>;

template <class T>
using TempValue = std::variant<T,Repeat>;

// helper type for the visitor #4
template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
// explicit deduction guide (not needed as of C++20)
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

template <typename T>
T get_value(const std::vector<uint8_t>& bytes, size_t pos){
  T* ptr = (T*) &bytes[pos];
  return *ptr;
}

template <typename T>
void set_value(std::vector<uint8_t>& bytes, size_t pos, T val){
  GetBytes<T> gb{val};
  for (unsigned long i_=0; i_<sizeof(T); i_++){
    bytes[pos+i_] = gb.bytes[i_];
  }
}

template <typename T>
void push_back(T arg, std::vector<uint8_t>& bytes, size_t& curr_count, bool& isValues, bool check = false){
  GetBytes<T> gb;
  gb.value = arg;

  uint16_t mask = (uint16_t)0x7FFF;
  uint16_t count = 0;
  if (check) {
    if (isValues && ((count = get_value<uint16_t>(bytes, curr_count)) < mask)) {
      auto temp_curr_count = curr_count;
      set_value<uint16_t>(bytes, curr_count, count + 1);
      push_back<T>(arg, bytes, curr_count, isValues, false);
      curr_count = temp_curr_count;
    } else {
      push_back<uint16_t>(1, bytes, curr_count, isValues, false);
      auto temp_curr_count = size_t(bytes.empty() ? 0 : bytes.size()-2);
      push_back<T>(arg, bytes, curr_count, isValues, false);
      curr_count = temp_curr_count;
    }
    isValues = true;
  } else {
    for (unsigned long i_=0; i_<sizeof(T); i_++){
      bytes.push_back(gb.bytes[i_]);
    }
    isValues = false;
    curr_count = 0;
  }
}

template <typename T>
std::vector<uint8_t> packLZ77_bytes(std::vector<TempValue<T>> vals){
  std::vector<uint8_t> bytes;
  size_t curr_count = 0;
  bool isValues = false;
  const auto runMask = (uint16_t)~0x7FFF;
  for (auto& val : vals){
    std::visit(overloaded {
            [&](T arg) { push_back(arg, bytes, curr_count, isValues, true); },
            [&](std::pair<uint16_t, uint16_t> arg) {
                push_back<uint16_t>(arg.second | runMask, bytes, curr_count, isValues);
                push_back<uint16_t>(arg.first, bytes, curr_count, isValues);
            }
    }, val);
  }
  return bytes;
}


template <typename T>
std::pair<std::vector<uint8_t>, int> packLZ77(std::vector<TempValue<T>> vals){
  std::vector<uint8_t> bytes = packLZ77_bytes(vals);
  return {bytes,bytes.size()};
//  int size = bytes.size();
//  while(bytes.size() % sizeof(T) != 0){
//    bytes.push_back(0);
//  }
//  T* bytes_data = (T*) bytes.data();
//  std::vector<T> values(bytes_data, bytes_data + (bytes.size() / sizeof(T)));
//
//  return {values, size};
}

Index makeLZ77Index(const std::vector<int>& rowptr) {
  return Index(lz77f,
               {ModeIndex({makeArray(rowptr)})});
}

/// Factory function to construct a compressed sparse row (CSR) matrix.
template<typename T>
TensorBase makeLZ77(const std::string& name, const std::vector<int>& dimensions,
                    const std::vector<int>& pos, const std::vector<uint8_t>& vals) {
  taco_uassert(dimensions.size() == 1);
  Tensor<T> tensor(name, dimensions, {LZ77});
  auto storage = tensor.getStorage();
  storage.setIndex(makeLZ77Index(pos));
  storage.setValues(makeArray(vals));
  tensor.setStorage(storage);
  return std::move(tensor);
}

Func getCopyFunc(){
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

Func getPlusFunc(){
  auto plusFunc = [](const std::vector<ir::Expr>& v) {
      return ir::Add::make(v[0], v[1]);
  };
  auto algFunc = [](const std::vector<IndexExpr>& v) {
      auto l = Region(v[0]);
      auto r = Region(v[1]);
      return Union(Union(l, r), Union(Complement(l), Complement(r)));
  };
  Func plus_("plus_", plusFunc, algFunc);
  return plus_;
}

template <typename T = uint8_t>
std::pair<Tensor<T>, std::vector<int>>
gen_random_lz77(std::string name, int size, double uncompressed_threshold,
                int lower_dist, int upper_dist,
                int lower_runs, int upper_runs,
                int lower_vals, int upper_vals) {
  std::uniform_int_distribution<int> unif_dist(lower_dist, upper_dist);
  std::uniform_int_distribution<int> unif_runs(lower_runs, upper_runs);
  std::uniform_int_distribution<int> unif_vals(lower_vals, upper_vals);
  std::uniform_real_distribution<double> unif_compressed(0,1);

  std::vector<TempValue<T>> vals;

//  int numRemaining = size-1;
//  while (numRemaining > 0) {
//    int run = std::min(USHRT_MAX, numRemaining-1);
//    vals.push_back((T) unif_vals(gen));
//    vals.push_back(Repeat{1, run});
//    numRemaining -= (run+1);
//  }

  int numRawValues = 1;
  int numRawPrevVals = 1;
  vals.push_back((T) unif_vals(gen));
  int numRemaining = size-2;
  while(numRemaining > 0){
    if (unif_compressed(gen) > uncompressed_threshold && numRemaining > 1 && numRawPrevVals > 0) {
      auto run = std::min(unif_runs(gen), numRemaining-1);
      auto dist_v = std::min(unif_dist(gen), numRawPrevVals);
      if (run == 0) continue;
      vals.push_back(Repeat{dist_v,run});
      numRawPrevVals = 0;
      numRemaining -= run;
    } else {
      vals.push_back((T) unif_vals(gen));
      numRawValues += 1;
      numRawPrevVals += 1;
      numRemaining -= 1;
    }
  }

  vals.push_back((T) unif_vals(gen));
  numRawValues += 1;

  auto packed = packLZ77<T>(vals);

  std::vector<int> data;
  data.push_back(packed.second);
  data.push_back(numRawValues);
  return {makeLZ77<T>("lz77_"+name, {size}, {0, packed.second}, packed.first), data};
}

constexpr int size_lower = 1'000'000; //1'000;
constexpr int size_upper = 1'000'000;
constexpr int size_mult  = 10;
constexpr int run_lower = 32;
constexpr int run_upper = 32; //65'536; //1'000'000;
constexpr int run_mult  = 16;

int numRandTensors = 0;
//constexpr int minElements = 10'000'000;

static void CustomArguments(benchmark::internal::Benchmark *b) {
  for (int size = size_lower; size <= size_upper; size *= size_mult) { // Size of vector
    for (int run = run_lower; run <= run_upper; run *= run_mult) {
      run = std::min(run, USHRT_MAX);
      b->Args({size, 0, run, true});
      for (int thresh = 0; thresh <= 10; thresh += 15) {
        b->Args({size, thresh, run, false});
      }
    }
  }
}

//static std::pair<Tensor<uint8_t>, Tensor<uint8_t>> getDense(Tensor<uint8_t> d0, Tensor<uint8_t> d1, int tsize){
//  Tensor<uint8_t> t0("t0_d", {tsize},   dv, 0);
//  Tensor<uint8_t> t1("t1_d", {tsize},   dv, 0);
//  const IndexVar i("i");
//  auto copy = getCopyFunc();
//
//  t0(i) = copy(d0(i));
//  t0.setAssembleWhileCompute(true);
//  t0.compile();
//  t0.compute();
//  d0 = t0;
//
//  t1(i) = copy(d1(i));
//  t1.setAssembleWhileCompute(true);
//  t1.compile();
//  t1.compute();
//  d1 = t1;
//
//  return {t0, t1};
//}

static void shim_compute(Tensor<TENSOR_TYPE> t){
  t.compute();
}

static void BM_all(benchmark::State &state) {
  int tsize = state.range(0);
  double thresh = state.range(1)/10.0;
  int run_upper = state.range(2);
  int run_lower = std::max(1, run_upper-100);
  bool isDense = state.range(3);
  auto plus_ = getPlusFunc();
  auto copy = getCopyFunc();

  auto vals_lower = std::numeric_limits<TENSOR_TYPE>::min();
  auto vals_upper = std::numeric_limits<TENSOR_TYPE>::max();

  auto [d0, v0] = gen_random_lz77<TENSOR_TYPE>("t0", tsize, thresh, 1, 50, run_lower, run_upper, vals_lower, vals_upper);
  auto [d1, v1] = gen_random_lz77<TENSOR_TYPE>("t1", tsize, thresh, 1, 50, run_lower, run_upper, vals_lower, vals_upper);

  if (isDense){
    Tensor<TENSOR_TYPE> t0("t0_d", {tsize},   dv, 0);
    Tensor<TENSOR_TYPE> t1("t1_d", {tsize},   dv, 0);
    const IndexVar i("i");

    t0(i) = copy(d0(i));
    t0.setAssembleWhileCompute(true);
    t0.compile();
    shim_compute(t0); //.compute();
    d0 = t0;

    t1(i) = copy(d1(i));
    t1.setAssembleWhileCompute(true);
    t1.compile();
    shim_compute(t1); //.compute();
    d1 = t1;
  }

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<TENSOR_TYPE> expected("expected_", {tsize}, isDense ? dv : lz77f);
    const IndexVar i("i");
    expected(i) = plus_(d0(i), d1(i));
    expected.setAssembleWhileCompute(true);
    expected.compile();
    state.ResumeTiming();

    expected.compute();
  }

  state.counters.insert({{"d0_bytes", v0[0]}, {"d1_bytes", v1[0]}, {"d0_raw_vals", v0[1]}, {"d1_raw_vals", v1[1]}});
}

static void BM_all_compute(benchmark::State &state) {
  int tsize = state.range(0);
  double thresh = state.range(1)/10.0;
  int run_upper = state.range(2);
  int run_lower = std::max(1, run_upper-100);
  bool isDense = state.range(3);
  auto plus_ = getPlusFunc();
  auto copy = getCopyFunc();

  auto vals_lower = std::numeric_limits<TENSOR_TYPE>::min();
  auto vals_upper = std::numeric_limits<TENSOR_TYPE>::max();

  auto [d0, v0] = gen_random_lz77<TENSOR_TYPE>("t0", tsize, thresh, 1, 50, run_lower, run_upper, vals_lower, vals_upper);
  auto [d1, v1] = gen_random_lz77<TENSOR_TYPE>("t1", tsize, thresh, 1, 50, run_lower, run_upper, vals_lower, vals_upper);

  if (isDense){
    Tensor<TENSOR_TYPE> t0("t0_d", {tsize},   dv, 0);
    Tensor<TENSOR_TYPE> t1("t1_d", {tsize},   dv, 0);
    const IndexVar i("i");

    t0(i) = copy(d0(i));
    t0.setAssembleWhileCompute(true);
    t0.compile();
    shim_compute(t0); //.compute();
    d0 = t0;

    t1(i) = copy(d1(i));
    t1.setAssembleWhileCompute(true);
    t1.compile();
    shim_compute(t1); //.compute();
    d1 = t1;
  }

  if(isDense) {
    for (auto _ : state) {
      state.PauseTiming();
      Tensor<TENSOR_TYPE> expected("expected_", {tsize}, isDense ? dv : lz77f);
      const IndexVar i("i");
      expected(i) = plus_(d0(i), d1(i));
      expected.setAssembleWhileCompute(true);
      expected.compile();
      state.ResumeTiming();

      expected.compute();
    }
  } else {
    Tensor<TENSOR_TYPE> expected("expected_", {tsize}, isDense ? dv : lz77f);
    taco_tensor_t* e = expected.getStorage();
    taco_tensor_t* l = d0.getStorage();
    taco_tensor_t* r = d1.getStorage();

    for (auto _ : state) {
      int res = compute(e, l, r);
    }
  }

  state.counters.insert({{"d0_bytes", v0[0]}, {"d1_bytes", v1[0]}, {"d0_raw_vals", v0[1]}, {"d1_raw_vals", v1[1]}});
}

BENCHMARK(BM_all_compute)->Apply(CustomArguments)\
                 ->MeasureProcessCPUTime()\
                 ->Unit(benchmark::kMicrosecond);
//                 ->Repetitions(10);

BENCHMARK_MAIN();