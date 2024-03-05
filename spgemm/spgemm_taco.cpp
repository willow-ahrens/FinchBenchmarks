#include "taco.h"
// #include "taco/format.h"
// #include "taco/lower/lower.h"
// #include "taco/ir/ir.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

using namespace taco;

extern int optind;

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  char *buf[100], *temp;
  temp = strtok(params.argv[0]," ");
  int buf_size=1;
  while (temp != NULL)
  {
    buf[buf_size++] = temp;
    temp = strtok(NULL, " ");
  }

  static struct option long_options[] = {
    {"help", no_argument, 0, 'h'},
    {"schedule", required_argument, 0, 's'},
    {"format_a", required_argument, 0, 'a'},
    {"format_b", required_argument, 0, 'b'},
    {0, 0, 0, 0}
  };

  std::string schedule = "gustavson";
  std::string format_a = "csr";
  std::string format_b = "csr";

  // Parse the options
  int option_index = 0;
  int c;
  optind = 1;
  while ((c = getopt_long(buf_size, buf, "hs:a:b:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        std::cout << "Options:" << std::endl;
        std::cout << "  -h, --help      Print this help message" << std::endl;
        std::cout << "  -s, --schedule  Execution schedule, from [gustavson, inner, outer]" << std::endl;
        std::cout << "  -a, --format_a  Format of A, from [csr, dcsr, dense]" << std::endl;
        std::cout << "  -b, --format_b  Format of B, from [csr, dcsr, dense]" << std::endl;
        exit(0);
      case 's':
        schedule = optarg;
        break;
      case 'a':
        format_a = optarg;
        break;
      case 'b':
        format_b = optarg;
        break;
      case '?':
        // getopt_long already printed an error message
        break;
      default:
        abort();
    }
  }

  // Check that all required options are present
  if (params.input.empty() || params.output.empty()) {
    std::cerr << "Missing required option" << std::endl;
    exit(1);
  }

  Tensor<double> A;
  if (format_a == "csr") {
    A = read(params.input+"/A.ttx", Format({Dense, Sparse}), true);
  } else if (format_a == "dcsr") {
    A = read(params.input+"/A.ttx", Format({Sparse, Sparse}), true);
  } else if (format_a == "dense") {
    A = read(params.input+"/A.ttx", Format({Dense, Dense}), true);
  } else {
    std::cerr << "Invalid format for A" << std::endl;
    exit(1);
  }
  Tensor<double> B;
  if (format_b == "csr") {
    B = read(params.input+"/B.ttx", Format({Dense, Sparse}), true);
  } else if (format_b == "dcsr") {
    B = read(params.input+"/B.ttx", Format({Sparse, Sparse}), true);
  } else if (format_b == "dense") {
    B = read(params.input+"/B.ttx", Format({Dense, Dense}), true);
  } else {
    std::cerr << "Invalid format for B" << std::endl;
    exit(1);
  }

  int m = A.getDimension(0);
  int n = B.getDimension(1);

  Tensor<double> C("C", {m, n}, Format({Dense, Sparse})); // cond


  IndexVar i, j, k;
  IndexStmt stmt;

  if (schedule == "gustavson") {
    C(i, j) += A(i, k) * B(k, j);
  } else if (schedule == "inner") {
    C(i, j) += A(i, k) * B(j, k);
    stmt= C.getAssignment().concretize();
    stmt = stmt.reorder({i,j,k}); 
  } else if (schedule == "outer") {
    C(i, j) += A(k, i) * B(k, j);
    stmt= C.getAssignment().concretize();
    stmt = stmt.reorder({k,i,j}); 
  } else {
    std::cerr << "Invalid schedule" << std::endl;
    exit(1);
  }

  C.compile();

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&C]() {
      C.setNeedsAssemble(true);
      C.setNeedsCompute(true);
    },
    [&C]() {
      C.assemble(); //no need for dense ouptut
      C.compute();
    }
  );

  write(params.output+"/C.ttx", C);

  if (params.verbose) {
    C.printAssembleIR(std::cout, true, true);
    C.printComputeIR(std::cout, true, true);
  }

  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output+"/measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
