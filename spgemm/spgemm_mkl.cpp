#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include <mkl.h>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"
#include <sys/time.h>
#include <xmmintrin.h>


struct EdgeList {
        MKL_INT src, dst;
        double val;
};

#define matrixA (0)
#define matrixB (1)

int n_rows_A, n_cols_A;
int n_rows_B, n_cols_B;
int n_rows_C, n_cols_C;
int nA0 = 1, nB0 = 1;
int nA1 = 2, nB1 = 2;

#define SIZE (128)
sparse_matrix_t A[SIZE], B[SIZE], C[SIZE];
sparse_matrix_t A0, B0;
sparse_matrix_t A1, B1;

int32_t nr, nc, ne;

MKL_INT *csr_row_pointerx[SIZE]; 
MKL_INT *csr_columnsAx[SIZE];
double *csr_valuesx[SIZE];
MKL_INT *csr_row_pointery[SIZE]; 
MKL_INT *csr_columnsAy[SIZE];
double *csr_valuesy[SIZE];

void flush_cache_line(void* ptr) {
    // Flushes the cache line that contains the given memory address (ptr)
    _mm_clflush(ptr);
}

void loadTTX(FILE *fp, int flag)
{
		char buf[1024];
		int nflag, sflag;
		int pre_count=0;
		long i;

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
MKL_INT *csr_row_pointer = (MKL_INT *)malloc(sizeof(MKL_INT)*(nr+1+SIZE));

 		for(int i=0;i<SIZE;i++) {
	       	       csr_columnsAx[i] = (MKL_INT *)malloc(sizeof(MKL_INT)*ne);
			csr_valuesx[i]  = (double *)malloc(sizeof(double)*ne);
			csr_row_pointerx[i] = (MKL_INT *)malloc(sizeof(MKL_INT)*(nr+1+SIZE));
	       	       csr_columnsAy[i] = (MKL_INT *)malloc(sizeof(MKL_INT)*ne);
			csr_valuesy[i]  = (double *)malloc(sizeof(double)*ne);
			csr_row_pointery[i] = (MKL_INT *)malloc(sizeof(MKL_INT)*(nr+1+SIZE));
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
				csr_columnsAx[j][idx] = csr_columnsA[idx] + j;
				csr_valuesx[j][idx] = csr_values[idx];
				csr_columnsAy[j][idx] = csr_columnsA[idx];//+ j;
				csr_valuesy[j][idx] = csr_values[idx];
				//csr_columnsAx[j][idx] = (csr_columnsA[idx]+j)%nc;
				//csr_valuesx[j][idx] = csr_values[idx]+j;
			}
		}

		for (uint32_t i = nr; i > 0; i--) {
			csr_row_pointer[i] = csr_row_pointer[i - 1];
		}
		csr_row_pointer[0] = 0;

		for(int i=0; i<=nr+SIZE; i++) {
			for(int j=0;j<SIZE;j++) {
				int i0 = i;
				if(i0 > nr) i0 = nr;
				csr_row_pointerx[j][i] = csr_row_pointer[i0];
				csr_row_pointery[j][i] = csr_row_pointer[i0];
			}
		}

		free(inputEdge);

		MKL_INT tmp_row_ptr[2] = {0,1};
		MKL_INT tmp_col_idx[2] = {0,0};
		double tmp_val[2] = {1.0, 0};
			mkl_sparse_d_create_csr(&A0, SPARSE_INDEX_BASE_ZERO, 1, 1,
					tmp_row_ptr, tmp_row_ptr+1,
					tmp_col_idx, tmp_val);
			mkl_sparse_d_create_csr(&B0, SPARSE_INDEX_BASE_ZERO, 1, 1,
					tmp_row_ptr, tmp_row_ptr+1,
					tmp_col_idx, tmp_val);
		int x1=2, x2=2;
		MKL_INT tmp_row_ptr2[3] = {0,2,4};
		MKL_INT tmp_row_end2[3] = {2,4};
		MKL_INT tmp_col_idx2[4] = {0,1,0,1};
		double tmp_val2[4] = {1.0,2.0,3.0,4.0};
			mkl_sparse_d_create_csr(&A1, SPARSE_INDEX_BASE_ZERO, x1, x2,
					tmp_row_ptr2, tmp_row_ptr2+1,
					tmp_col_idx2, tmp_val2);
			mkl_sparse_d_create_csr(&B1, SPARSE_INDEX_BASE_ZERO, x1, x2,
					tmp_row_ptr2, tmp_row_ptr2+1,
					tmp_col_idx2, tmp_val2);

	
			if(flag == matrixA) {
			n_rows_A = nr;
			n_cols_A = nc;
			for(int i=0;i<SIZE;i++) {
				int nrx = nr+i, ncx = nc+i;
			mkl_sparse_d_create_csr(&A[i], SPARSE_INDEX_BASE_ZERO, nrx, ncx,
					csr_row_pointerx[i], csr_row_pointerx[i] + 1,
					csr_columnsAx[i], csr_valuesx[i]);
			}
		} else if(flag == matrixB) {
			n_rows_B = nr;
			n_cols_B = nc;
			for(int i=0;i<SIZE;i++) {
				int nrx = nr+i, ncx = nc+i;
			mkl_sparse_d_create_csr(&B[i], SPARSE_INDEX_BASE_ZERO, nrx, ncx,
					csr_row_pointerx[i], csr_row_pointerx[i] + 1,
					csr_columnsAx[i], csr_valuesx[i]);
			}
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

  int ix=0;
  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;

//fprintf(stderr, "OK");
  MKL_INT *rows_start[SIZE+1], *rows_end[SIZE+1], *columns[SIZE+1];
  double *values[SIZE+1];
  long sum=0;
  int r;
  // Assemble output indices and numerically compute the result
  auto time = benchmark(
		  [&sum, &r, &ix, &indexing, &rows_start, &rows_end, &columns, &values]() {
    },
    [&sum, &r, &ix, &indexing, &rows_start, &rows_end, &columns, &values]() {

    matrix_descr descrC;
    descrC.type = SPARSE_MATRIX_TYPE_GENERAL;

    matrix_descr descrA, descrB;
    descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
    //r = rand()%SIZE;
    r = 0;
		rows_start[r] = NULL;
		rows_end[r] = NULL;
		columns[r] = NULL;
		values[r] = NULL;
		int nrx = nr+r, ncx = nc+r;
		  sparse_matrix_t A, B, C, C0, C1;
		  mkl_sparse_d_create_csr(&A, SPARSE_INDEX_BASE_ZERO, nrx, ncx,
				  csr_row_pointerx[r], csr_row_pointerx[r] + 1,
				  csr_columnsAx[r], csr_valuesx[r]);
		  mkl_sparse_d_create_csr(&B, SPARSE_INDEX_BASE_ZERO, nrx, ncx,
				  csr_row_pointery[r], csr_row_pointery[r] + 1,
				  csr_columnsAy[r], csr_valuesy[r]);

		/*
		  const size_t array_size = 1024 * 1024 * 32;  // 1MB array
		  int* data = new int[array_size];        // Dynamically allocate an array
		  for (size_t i = 0; i < array_size; ++i) {
			  data[i] = i;
		  }
		  for (size_t i = 0; i < array_size; ++i) {
			  flush_cache_line(&data[i]);  // Flush each cache line
		  }
		  for (size_t i = 0; i < array_size; ++i) {
			  sum +=  data[i];
		  }*/


		  //mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A0, B0, &C0);
		  //mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A1, B1, &C1);
#ifdef TESTT
struct timeval start, end;
    long seconds, useconds;
    double duration;

    gettimeofday(&start, NULL);
#endif
//		  mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A, B, &C);
//		  mkl_sparse_d_export_csr(C, &indexing, &nrx, &ncx, &rows_start[r], &rows_end[r], &columns[r], &values[r]);



	   C = NULL;
    //mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A, SPARSE_OPERATION_NON_TRANSPOSE,descrB, B, SPARSE_STAGE_NNZ_COUNT, &C);
    //mkl_sparse_d_export_csr(C, &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
    mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A, SPARSE_OPERATION_NON_TRANSPOSE,descrB, B, SPARSE_STAGE_FULL_MULT, &C);
mkl_sparse_order(C);
    mkl_sparse_d_export_csr(C, &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
#ifdef TESTT
		  gettimeofday(&end, NULL); 
   seconds = end.tv_sec - start.tv_sec;
    useconds = end.tv_usec - start.tv_usec;
    duration = seconds + useconds / 1e6;
//fprintf(stderr, "(time: %d %f)\n", r, duration);
fprintf(stderr, "(time: %d %d %f)\n", r, rows_start[r][nr+r], duration);
#endif
		  //fprintf(stderr, "%p %p %p %p\n", rows_start[r], rows_end[r], columns[r], values[r]);


		  //sparse_matrix_t C0, C;
		  //mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A0, B0, &C0);
		  //mkl_sparse_d_export_csr(C0, &indexing, &nA0, &nB0, &rows_start[SIZE], &rows_end[SIZE], &columns[SIZE], &values[SIZE]);

	  //mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A[r], B[r], &C[r]);
	  //mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
    /*
    C[r] = NULL;
		  mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A0, SPARSE_OPERATION_NON_TRANSPOSE,descrB, B0, SPARSE_STAGE_NNZ_COUNT, &C[r]);
		  mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[SIZE], &rows_end[SIZE], &columns[SIZE], &values[SIZE]);
		  mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A0, SPARSE_OPERATION_NON_TRANSPOSE,descrB, B0, SPARSE_STAGE_FINALIZE_MULT, &C[r]);
		  mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[SIZE], &rows_end[SIZE], &columns[SIZE], &values[SIZE]);
*/


/*

    if(ix % 2) {
	    C[r] = NULL;
    mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A[r], SPARSE_OPERATION_NON_TRANSPOSE,descrB, B[r], SPARSE_STAGE_NNZ_COUNT, &C[r]);
    mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
    mkl_sparse_sp2m (SPARSE_OPERATION_NON_TRANSPOSE, descrA, A[r], SPARSE_OPERATION_NON_TRANSPOSE,descrB, B[r], SPARSE_STAGE_FINALIZE_MULT, &C[r]);
    mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
    } else {
	    C[r] = NULL;
		  mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, A[r], B[r], &C[r]);
		  mkl_sparse_d_export_csr(C[r], &indexing, &n_rows_A, &n_cols_B, &rows_start[r], &rows_end[r], &columns[r], &values[r]);
    }
ix++;
*/		
		
      	  //mkl_sparse_destroy(C);
      	  //mkl_sparse_destroy(C0);
	 
		 mkl_sparse_destroy(A);
	 	 mkl_sparse_destroy(B);
		 ////mkl_finalize();
		 ////mkl_free_buffers();
	 	 //mkl_sparse_destroy(C);
	 	 //mkl_sparse_destroy(C0);
	 	 //mkl_sparse_destroy(C1);
    }
  );



//fprintf(stderr, "ERR0\n");
  FILE *fpC = fopen((params.output+"/C.ttx").c_str(), "w");

  fprintf(fpC, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(fpC, "%d %d %d\n", n_rows_A+r, n_cols_B+r, rows_start[r][n_rows_A]);

  for(int i=0;i<n_rows_A;i++) {
	for(int j=rows_start[0][i]; j<rows_start[r][i+1]; j++) {
		fprintf(fpC, "%d %d %lf\n", i+1, columns[r][j]+1, values[r][j]);
	}
  }

  fclose(fpC);

//fprintf(stderr, "ERR\n");
  json measurements;
  measurements["time"] = time;
  measurements["memory"] = 0;
  std::ofstream measurements_file(params.output+"/measurements.json");
  measurements_file << measurements;
  measurements_file.close();
  return 0;
}
