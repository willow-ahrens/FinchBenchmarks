#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <mkl.h>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

struct EdgeList {
	MKL_INT src, dst;
	double val;
};  

int n_rows_A;

sparse_matrix_t A;


double *loadXd(FILE *fp)
{
	char buf[1024];
	int n;

	while (fgets(buf, sizeof(buf), fp)) {
		if (buf[0] != '%') {
			break;
		}
	}
	sscanf(buf, "%d", &n);
	double *x = (double *)malloc(sizeof(double)*n);

	for(int i=0;i<n;i++) {
		int idx; double val;
		fscanf(fp, "%d %lf", &idx, &val);
		x[idx-1] = val;
	}
	return x;
}


void loadTTX(FILE *fp)
{
		char buf[1024];
		int nflag, sflag;
		int pre_count=0;
		long i;
		int32_t nr, nc, ne;

		fgets(buf, 1024, fp);
		if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; 
		else sflag = 0;
		if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
		else if(strstr(buf, "complex") != NULL) nflag = -1;
		else nflag = 1;
#ifdef SYM
		sflag = 1;
#endif

		while(1) {
			pre_count++;
			fgets(buf, 1024, fp);
			if(strstr(buf, "%") == NULL) break;
		}


		sscanf(buf, "%d %d %d", &nr, &nc, &ne);

		n_rows_A = nr;

		ne *= (sflag+1);

		EdgeList *inputEdge = (EdgeList *)malloc(sizeof(EdgeList)*(ne+1));

		for(i=0;i<ne;i++) {
			fscanf(fp, "%d %d", &inputEdge[i].src, &inputEdge[i].dst);
			inputEdge[i].src--; inputEdge[i].dst--;

			if(inputEdge[i].src < 0 || inputEdge[i].src >= nr || inputEdge[i].dst < 0 || inputEdge[i].dst >= nc) {
				fprintf(stdout, "A vertex id is out of range %d %d\n", inputEdge[i].src, inputEdge[i].dst);
				exit(0);
			}

			if(nflag == 1) {
				double ftemp;
				fscanf(fp, " %lf ", &ftemp);
				inputEdge[i].val = ftemp;
			} else if(nflag == -1) { // complex
				double ftemp1, ftemp2;
				fscanf(fp, " %lf %lf ", &ftemp1, &ftemp2);
				inputEdge[i].val = ftemp1;
			}

			if(sflag == 1) {
				i++;
				inputEdge[i].src = inputEdge[i-1].dst;
				inputEdge[i].dst = inputEdge[i-1].src;
				inputEdge[i].val = inputEdge[i-1].val;
			}
		}
		std::sort(inputEdge, inputEdge+ne, [](EdgeList x, EdgeList y) {
				if(x.src < y.src) return true;
				else if(x.src > y.src) return false;
				else return (x.dst < y.dst);
				});

		EdgeList *unique_end = std::unique(inputEdge, inputEdge + ne, [](EdgeList x, EdgeList y) {
				return x.src == y.src && x.dst == y.dst;
				});
		ne = unique_end - inputEdge;

		double *csr_values = (double *)malloc(sizeof(double)*ne);
		MKL_INT *csr_columns = (MKL_INT *)malloc(sizeof(MKL_INT)*ne);
		MKL_INT *csr_row_pointer = (MKL_INT *)malloc(sizeof(MKL_INT)*(nr+1));

		for (uint32_t i = 0; i <= nr; i++) {
			csr_row_pointer[i] = 0;
		}

		for (uint32_t i = 0; i < ne; i++) {
			csr_row_pointer[inputEdge[i].src + 1]++;
		}

		for (uint32_t i = 1; i <= nr; i++) {
			csr_row_pointer[i] += csr_row_pointer[i - 1];
		}

		for (uint32_t i = 0; i < ne; i++) {
			uint32_t src = inputEdge[i].src;
			uint32_t idx = csr_row_pointer[src]++;
			csr_columns[idx] = inputEdge[i].dst;
			csr_values[idx] = inputEdge[i].val;
		}

		for (uint32_t i = nr; i > 0; i--) {
			csr_row_pointer[i] = csr_row_pointer[i - 1];
		}
		csr_row_pointer[0] = 0;

		free(inputEdge);

		mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nr, nc,
				csr_row_pointer, csr_row_pointer + 1,
				csr_columns, csr_values);
}

int main(int argc, char **argv) {
  auto params = parse(argc, argv);

  FILE *fpA = fopen((params.input+"/A.ttx").c_str(), "r");
  FILE *fpB = fopen((params.input+"/x.ttx").c_str(), "r");
  
   mkl_set_num_threads(1);

 loadTTX(fpA);
  double *x = loadXd(fpB);
  double *y = (double *)malloc(sizeof(double)*n_rows_A);
  fclose(fpA);
  fclose(fpB);


  struct matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL; 


  // Assemble output indices and numerically compute the result
  auto time = benchmark(
    [&x, &y, &descr]() {
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
    },
    [&x, &y, &descr]() {
	mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, A, descr, x, 0.0, y);
    }
  );




  FILE *fpC = fopen((params.input+"/y.ttx").c_str(), "w");

  fprintf(fpC, "%%%%MatrixMarket tensor array real general\n");
  fprintf(fpC, "%d\n", n_rows_A);

  for (int k = 0; k < n_rows_A; ++k) {
	  fprintf(fpC, "%lf\n", y[k]);
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
