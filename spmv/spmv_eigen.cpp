#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <Eigen/Sparse>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

Eigen::VectorXd loadXd(FILE *fp)
{
	char buf[1024];
	int n;

	while (fgets(buf, sizeof(buf), fp)) {
		if (buf[0] != '%') {
			break;
		}
	}
	sscanf(buf, "%d", &n);

	Eigen::VectorXd x(n);
	
	for(int i=0;i<n;i++) {
		int idx; double val;
		fscanf(fp, "%d %lf", &idx, &val);
		x(idx-1) = val;
	}
	return x;
}


Eigen::SparseMatrix<double> loadTTX(FILE *fp)
{
	char buf[1024];
	int nr, nc, nnz;

	while (fgets(buf, sizeof(buf), fp)) {
		if (buf[0] != '%') {
			break;
		}
	}
	sscanf(buf, "%d %d %d", &nr, &nc, &nnz);
	std::vector<Eigen::Triplet<double>> tripletList;
	tripletList.reserve(nnz);
	for(int i=0;i<nnz;i++) {
		int row, col; double val;
		fscanf(fp, "%d %d %lf", &row, &col, &val);
		tripletList.emplace_back(row - 1, col - 1, val);
	}
	Eigen::SparseMatrix<double> mat(nr, nc);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	mat.makeCompressed(); // Ensure the matrix is in CSR format
	return mat;
}

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  FILE *fpA = fopen((params.input+"/A.ttx").c_str(), "r");
  FILE *fpB = fopen((params.input+"/x.ttx").c_str(), "r");
  
  Eigen::SparseMatrix<double> A = loadTTX(fpA);
  Eigen::VectorXd x = loadXd(fpB);
  Eigen::VectorXd y;
  fclose(fpA);
  fclose(fpB);

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&A, &x, &y]() {
	y = A * x;
    },
    [&A, &x, &y]() {
	y = A * x;
    }
  );

  FILE *fpC = fopen((params.input+"/y.ttx").c_str(), "w");

  fprintf(fpC, "%%%%MatrixMarket tensor array real general\n");
  fprintf(fpC, "%ld\n", y.size());

  for (int k = 0; k < y.size(); ++k) {
	  fprintf(fpC, "%lf\n", y(k));
  }

  fclose(fpC);

  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output+"/measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
