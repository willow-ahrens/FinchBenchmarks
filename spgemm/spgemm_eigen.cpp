#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <Eigen/Sparse>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

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
		//tripletList.emplace_back(row, col, val);
	}
	Eigen::SparseMatrix<double> mat(nr, nc);
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	mat.makeCompressed(); // Ensure the matrix is in CSR format
	return mat;
}

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  FILE *fpA = fopen((params.input+"/A.ttx").c_str(), "r");
  FILE *fpB = fopen((params.input+"/B.ttx").c_str(), "r");
  
  Eigen::SparseMatrix<double> A = loadTTX(fpA);
  Eigen::SparseMatrix<double> B = loadTTX(fpB);
  Eigen::SparseMatrix<double> C;
  fclose(fpA);
  fclose(fpB);

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&A, &B, &C]() {
	C = A * B;
    },
    [&A, &B, &C]() {
	C = A * B;
    }
  );

  FILE *fpC = fopen((params.output+"/C.ttx").c_str(), "w");

fprintf(fpC, "%%%%MatrixMarket matrix coordinate real general\n");
fprintf(fpC, "%ld %ld %ld\n", C.rows(), C.cols(), C.nonZeros());

  for (int k = 0; k < C.outerSize(); ++k) {
	  for (Eigen::SparseMatrix<double>::InnerIterator it(C, k); it; ++it) {
		  fprintf(fpC, "%ld %ld %lf\n", it.row()+1, it.col()+1, it.value());
	  }
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
