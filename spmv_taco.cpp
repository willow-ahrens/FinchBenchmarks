#include "taco/tensor.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <getopt.h>
#include <iostream>
#include <string>

template <typename Setup, typename Test>
double benchmark(double time_max, int trial_max, Setup setup, Test test) {
  auto time_total = std::chrono::high_resolution_clock::duration(0);
  auto time_min = std::chrono::high_resolution_clock::duration(0);
  int trial = 0;
  while (trial < trial_max) {
    setup();
    auto tic = std::chrono::high_resolution_clock::now();
    test();
    auto toc = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(toc - tic);
    trial++;
    if (trial == 1 || time < time_min) {
      time_min = time;
    }
    time_total += time;
    if (time_total.count() * 1e-9 > time_max) {
      break;
    }
  }
  return time_min.count() * 1e-9;
}

using namespace taco;

int main(int argc, char **argv) {
  long n_trials = 10000; // max number of trials
  double t_trials = 10.0; // max trial time in seconds
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

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
      t_trials, n_trials,
      [&y]() {
        y.assemble();
        // tensor_a.setNeedsCompute(true);
      },
      [&y]() { y.compute(); });

  std::cout << time << std::endl;

  write(file_x, x);

  return 0;
}