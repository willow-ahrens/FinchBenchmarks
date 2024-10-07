#include <fstream>
#include <vector>
#include <iostream>
#include <cublas_v2.h>
#include "util.h"
#include "kernel.h"

#define CEIL(A, F) (F * ((A + F - 1) / (F)))

double runBatch(int batch_size, int* M, int* N, int* K, int iters, bool warmup, int TLP_thres) {
  int BATCH = batch_size;
  float **A;
  float **B;
  float **C;

  A = (float**) malloc(BATCH * sizeof(float*));
  B = (float**) malloc(BATCH * sizeof(float*));
  C = (float**) malloc(BATCH * sizeof(float*));

  for (int i=0; i<BATCH; ++i){
    ErrChk(cudaMalloc((void**)&A[i], CEIL(M[i], 256)*CEIL(K[i], 256)*sizeof(float)));
    ErrChk(cudaMalloc((void**)&B[i], CEIL(K[i], 256)*CEIL(N[i], 256)*sizeof(float)));
    ErrChk(cudaMalloc((void**)&C[i], CEIL(M[i], 256)*CEIL(N[i], 256)*sizeof(float)));
  }

  float **dev_A;
  float **dev_B;
  float **dev_C;

  ErrChk(cudaMalloc((void**)&dev_A, BATCH*sizeof(float*)));
  ErrChk(cudaMalloc((void**)&dev_B, BATCH*sizeof(float*)));
  ErrChk(cudaMalloc((void**)&dev_C, BATCH*sizeof(float*)));

  ErrChk(cudaMemcpy(dev_A, A, BATCH*sizeof(float*), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(dev_B, B, BATCH*sizeof(float*), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(dev_C, C, BATCH*sizeof(float*), cudaMemcpyHostToDevice));


  int *dev_M, *dev_N, *dev_K;
  ErrChk(cudaMalloc((void**)&dev_M, BATCH*sizeof(int)));
  ErrChk(cudaMalloc((void**)&dev_N, BATCH*sizeof(int)));
  ErrChk(cudaMalloc((void**)&dev_K, BATCH*sizeof(int)));

  ErrChk(cudaMemcpy(dev_M, M, BATCH*sizeof(int), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(dev_N, N, BATCH*sizeof(int), cudaMemcpyHostToDevice));
  ErrChk(cudaMemcpy(dev_K, K, BATCH*sizeof(int), cudaMemcpyHostToDevice));


  float elapsedTime = 0.f;
  double time=0.f;
  float gflops_per_sec = 0.f;
  double gflops = 0.f;
  for (int i=0; i<BATCH; ++i)
    gflops += ((2 * int64_t(M[i]) * int64_t(N[i]) * int64_t(K[i])) + (2 * int64_t(M[i]) * int64_t(N[i])) ) / 1.0e9;
  cudaEvent_t start, stop;

  //Tiling Strategy
  int TLP = 0;

  const int tile_size[6][2] = {
    16, 16,
    32, 32,
    64, 64,
    128, 64,
    64, 128,
    128, 128
  };

  int *t_strategy;
  t_strategy = (int*) malloc(BATCH * sizeof(int));

  int t;
  for (t=0; t<6; ++t){
    TLP = 0;
    for (int j=0; j<BATCH; ++j)
      TLP += (M[j]/tile_size[t][0])*(N[j]/tile_size[t][1])*256;

    if (TLP < TLP_thres)
      break;
  }

  for (int j=0; j<BATCH; ++j){

    t_strategy[j] = 0;
    t = (t==6?5:t);

    if (tile_size[t][0] <= M[j] && tile_size[t][1] <= N[j])
      t_strategy[j] = t;
    else{
      for (int k=0; k<t; ++k){
	if (tile_size[k][0] == M[j] && tile_size[k][1] <= N[j]){
	  t_strategy[j] = k;
	}
      }
    }
  }



  int *dev_T;
  ErrChk(cudaMalloc((void**)&dev_T, BATCH*sizeof(int)));
  ErrChk(cudaMemcpy(dev_T, t_strategy, BATCH*sizeof(int), cudaMemcpyHostToDevice));


  //print the obtained tiling strategy
  // for (int j=0; j<BATCH; ++j) {
  //   printf("%d ", t_strategy[j]);
  // }
  // printf("\n");

  //Batching Strategy
  int *b_strategy;
  b_strategy = (int*) malloc(BATCH * sizeof(int));

  for (int j=0; j<BATCH; ++j){
    b_strategy[j] = 1;
  }

  for (int j=0; j<BATCH; ++j){
    TLP -= M[j]/2/tile_size[t_strategy[j]][0]*N[j]/tile_size[t_strategy[j]][1];
    if (TLP > TLP_thres && M[j]>t_strategy[j] && K[j]<=32)
      b_strategy[j] = 2;
  }


  int *dev_Ba;
  ErrChk(cudaMalloc((void**)&dev_Ba, BATCH*sizeof(int)));
  ErrChk(cudaMemcpy(dev_Ba, b_strategy, BATCH*sizeof(int), cudaMemcpyHostToDevice));



  //print the obtained batching strategy
  // for (int j=0; j<BATCH; ++j) {
  //   printf("%d ", b_strategy[j]);
  // }
  // printf("\n");


  //GEMM
  dim3 block_size;
  block_size.x = 256;
  block_size.y = 1;
  block_size.z = 1;

  dim3 grid_size;

  grid_size.x = M[0]/b_strategy[0]/tile_size[t_strategy[0]][0];
  grid_size.y = N[0]/b_strategy[0]/tile_size[t_strategy[0]][1];
  grid_size.z = BATCH;
  for (int j=1; j<BATCH; ++j){
    grid_size.x = (grid_size.x > M[j]/b_strategy[j]/tile_size[t_strategy[j]][0])? (grid_size.x):(M[j]/b_strategy[j]/tile_size[t_strategy[j]][0]);
    grid_size.y = (grid_size.y > N[j]/tile_size[t_strategy[j]][1])? (grid_size.y):(N[j]/tile_size[t_strategy[j]][1]);
  }

  //	printf("%d %d %d\n", grid_size.x, grid_size.y, grid_size.z);

  //warm-up
  if (warmup) {
    gemm_256<<<grid_size, block_size, sizeof(float)*4*128*8>>>(dev_M, dev_N, dev_K, dev_A, dev_B, dev_C, dev_T, dev_Ba);
    KernelErrChk();
  }

  ErrChk(cudaEventCreate(&start));
  ErrChk(cudaEventRecord(start,0));

  for (int run = 0; run < iters; ++run){
    gemm_256<<<grid_size, block_size, sizeof(float)*4*128*8>>>(dev_M, dev_N, dev_K, dev_A, dev_B, dev_C, dev_T, dev_Ba);
    KernelErrChk();
  }

  ErrChk(cudaEventCreate(&stop));
  ErrChk(cudaEventRecord(stop,0));
  ErrChk(cudaEventSynchronize(stop));
  ErrChk(cudaEventElapsedTime(&elapsedTime, start,stop));

  time = elapsedTime/iters;
  // printf("Execution time: %f\n", time);
  // time /= 1.0e3; //convert time unit from millisecond to second
  // gflops_per_sec   = gflops / time;
  // printf("GLOPS/sec: %f\n", gflops_per_sec);

  for (int i=0; i<BATCH; ++i){
    ErrChk(cudaFree(A[i]));
    ErrChk(cudaFree(B[i]));
    ErrChk(cudaFree(C[i]));
  }

  free(A);
  free(B);
  free(C);
  free(t_strategy);

  ErrChk(cudaFree(dev_M));
  ErrChk(cudaFree(dev_N));
  ErrChk(cudaFree(dev_K));
  ErrChk(cudaFree(dev_T));

  ErrChk(cudaFree(dev_A));
  ErrChk(cudaFree(dev_B));
  ErrChk(cudaFree(dev_C));

  return time;
}


int main(int argc, char** argv) {
  int NUM_HEADS = 8;
  int HEAD_SIZE = 64;

  ErrChk(cudaSetDevice(0));

  if(argc < 6){
    printf("Usage: input the batch size and the data file\n");
    exit(EXIT_FAILURE);
  }

  int batch_size = std::stoi(argv[1]);
  int num_batches = std::stoi(argv[2]);
  std::string data_file = argv[3];
  int iters = std::stoi(argv[4]);
  int warmup = std::stoi(argv[5]);
  std::string mode = argv[6];
  //int TLP_thres = atoi(argv[7]);
  int TLP_thres = 65536*2;

  batch_size *= ((mode == "qkt" || mode == "attn_v") ? NUM_HEADS : 1);

  // std::cout << "Opening file " << data_file << std::endl;

  std::fstream fs;
  fs.open(data_file);
  if (!fs.is_open()){
    printf("Error opening input\n");
    exit(EXIT_FAILURE);
  }

  double total_time = 0;
  for (int b = 0; b < num_batches; ++b) {
    std::vector<int> Ms(batch_size, -1);
    std::vector<int> Ns(batch_size, -1);
    std::vector<int> Ks(batch_size, -1);

    // std::cout << "BS " << batch_size << std::endl;

    if (mode == "gemm") {
      //read matrix config
      for (int i = 0; i < batch_size; ++i){
	fs >> Ms[i] >> Ns[i] >> Ks[i];
	Ms[i] = CEIL(Ms[i], 16);
	Ns[i] = CEIL(Ns[i], 16);
	Ks[i] = CEIL(Ks[i], 16);
      }
    } else {
      for (int i = 0; i < (batch_size / NUM_HEADS); ++i){
	int length;
	fs >> length;
	for (int j = 0; j < NUM_HEADS; ++j) {
	  int n = 16;
	  length = n * ((length + (n - 1)) / n);
	  if (mode == "qkt") {
	    Ms[i*8+j] = length;
	    Ns[i*8+j] = length;
	    Ks[i*8+j] = HEAD_SIZE;
	  } else if (mode == "attn_v") {
	    Ms[i*8+j] = length;
	    Ns[i*8+j] = HEAD_SIZE;
	    Ks[i*8+j] = length;
	  } else if (mode == "bp_test") {
	    Ms[i*8+j] = length;
	    Ns[i*8+j] = 512;
	    Ks[i*8+j] = 512;
	  } else {
	    std::cout << "No such mode" << std::endl;
	    exit(-1);
	  }
	}
      }
    }

    total_time += runBatch(batch_size, Ms.data(), Ns.data(), Ks.data(), iters, warmup, TLP_thres);
  }

  total_time /= num_batches;

  std::cout << "RESULTS," << total_time << std::endl;
  return 0;
}
