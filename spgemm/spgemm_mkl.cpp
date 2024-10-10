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

#define matrixA (0)
#define matrixB (1)

int n_rows_A, n_cols_A;
int n_rows_B, n_cols_B;
int n_rows_C, n_cols_C;

#define SIZE (1)
sparse_matrix_t A[SIZE], B[SIZE], C[SIZE];

void loadTTX(FILE *fp, int flag)
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
		MKL_INT *csr_columnsA = (MKL_INT *)malloc(sizeof(MKL_INT)*ne);
		MKL_INT *csr_columnsB = (MKL_INT *)malloc(sizeof(MKL_INT)*ne);
		MKL_INT *csr_row_pointer = (MKL_INT *)malloc(sizeof(MKL_INT)*(nr+1));
		MKL_INT *csr_row_pointerx[SIZE]; 

		MKL_INT *csr_columnsAx[SIZE];
		double *csr_valuesx[SIZE];
	       for(int i=0;i<SIZE;i++) {
	       	       csr_columnsAx[i] = (MKL_INT *)malloc(sizeof(MKL_INT)*ne);
			csr_valuesx[i]  = (double *)malloc(sizeof(double)*ne);
			csr_row_pointerx[i] = (MKL_INT *)malloc(sizeof(MKL_INT)*(nr+1));
	       }

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
			csr_columnsA[idx] = inputEdge[i].dst;
			csr_values[idx] = inputEdge[i].val;
			for(int j=0;j<SIZE;j++) {
				csr_columnsAx[j][idx] = (csr_columnsA[idx]+j)%nc;
				csr_valuesx[j][idx] = csr_values[idx]+j;
			}
		}

		for (uint32_t i = nr; i > 0; i--) {
			csr_row_pointer[i] = csr_row_pointer[i - 1];
		}
		csr_row_pointer[0] = 0;

		for(int i=0; i<=nr; i++) {
			for(int j=0;j<SIZE;j++) {
				csr_row_pointerx[j][i] = csr_row_pointer[i];
			}
		}

		free(inputEdge);

		if(flag == matrixA) {
			n_rows_A = nr;
			n_cols_A = nc;
			for(int i=0;i<SIZE;i++) 
			mkl_sparse_d_create_csr(&A[i], SPARSE_INDEX_BASE_ZERO, nr, nc,
					csr_row_pointerx[i], csr_row_pointerx[i] + 1,
					csr_columnsAx[i], csr_valuesx[i]);
		} else if(flag == matrixB) {
			n_rows_B = nr;
			n_cols_B = nc;
			for(int i=0;i<SIZE;i++)
			mkl_sparse_d_create_csr(&B[i], SPARSE_INDEX_BASE_ZERO, nr, nc,
					csr_row_pointerx[i], csr_row_pointerx[i] + 1,
					csr_columnsAx[i], csr_valuesx[i]);
		}
}


int main(int argc, char **argv) {

 mkl_set_num_threads(1);	
  auto params = parse(argc, argv);

  FILE *fpA = fopen((params.input+"/A.ttx").c_str(), "r");
  FILE *fpB = fopen((params.input+"/B.ttx").c_str(), "r");
  
  loadTTX(fpA, matrixA);
  loadTTX(fpB, matrixB);

  fclose(fpA);
  fclose(fpB);

  srand(time(NULL));

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;


  MKL_INT *rows_start[SIZE], *rows_end[SIZE], *columns[SIZE];
  double *values[SIZE];

  // Assemble output indices and numerically compute the result
  auto time = benchmark(
		  [&indexing, &rows_start, &rows_end, &columns, &values]() {

		  matrix_descr descrC;
		  descrC.type = SPARSE_MATRIX_TYPE_GENERAL;

		  matrix_descr descrA, descrB;
		  descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
		  descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
		  int r = rand()%SIZE;
		  C[r] = NULL;
		  mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A[r], SPARSE_OPERATION_NON_TRANSPOSE,descrB, B[r], SPARSE_STAGE_NNZ_COUNT, &C[r]);
		  mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
		  mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A[r], SPARSE_OPERATION_NON_TRANSPOSE,descrB, B[r], SPARSE_STAGE_FINALIZE_MULT, &C[r]);
		  mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);

		  mkl_sparse_destroy(C[r]);
    },
    [&indexing, &rows_start, &rows_end, &columns, &values]() {

    matrix_descr descrC;
    descrC.type = SPARSE_MATRIX_TYPE_GENERAL;

    matrix_descr descrA, descrB;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
    int r = rand()%SIZE;
    C[r] = NULL;
    mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A[r], SPARSE_OPERATION_NON_TRANSPOSE,descrB, B[r], SPARSE_STAGE_NNZ_COUNT, &C[r]);
    mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
    mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A[r], SPARSE_OPERATION_NON_TRANSPOSE,descrB, B[r], SPARSE_STAGE_FINALIZE_MULT, &C[r]);
    mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);

    mkl_sparse_destroy(C[r]);
    }
  );




  FILE *fpC = fopen((params.output+"/C.ttx").c_str(), "w");

  fprintf(fpC, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fpC, "%d %d %d\n", n_rows_A, n_cols_B, rows_start[0][n_rows_A]);

  for(int i=0;i<n_rows_A;i++) {
	for(int j=rows_start[0][i]; j<rows_start[0][i+1]; j++) {
		fprintf(fpC, "%d %d %lf\n", i+1, columns[0][j]+1, values[0][j]);
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
