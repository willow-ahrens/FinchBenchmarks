#include <cublas_v2.h>
#include <iostream>
#include <assert.h>
#include <cuda.h>

#include "utils.h"

int find_max(std::vector<int> v) {
  int max = -10000;
  for (auto e: v) {
    if (e > max) max = e;
  }
  return max;
}

float testCuBLAS(int batch_size, std::vector<int> ms, std::vector<int> ns, std::vector<int> ks,
		 std::string mode, int iters, int warmup) {
  int maxM = find_max(ms);
  int maxN = find_max(ns);
  int maxK = find_max(ks);

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);

  float* A;
  float* B;
  float* C;

  auto op_a = mode[0] == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;
  auto op_b = mode[1] == 'n' ? CUBLAS_OP_N : CUBLAS_OP_T;

  int lda = (op_a == CUBLAS_OP_N) ? maxM : maxK;
  int ldb = (op_b == CUBLAS_OP_N) ? maxK : maxN;
  int ldc = maxM;

  long long int strideA = maxM * maxK;
  long long int strideB = maxK * maxN;
  long long int strideC = maxM * maxN;

  CUDA_CHECK(cudaMalloc((void**)&A, batch_size * maxM * maxK * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&B, batch_size * maxK * maxN * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void**)&C, batch_size * maxM * maxN * sizeof(float)));

  auto runner = [&]() {
    float time = 0;
    for (int i = 0; i < iters; ++i) {
      cudaEvent_t start, end;
      float elapsed = 0;

      // Timing info
      cudaEventCreate(&start);
      cudaEventCreate(&end);
      cudaEventRecord(start);

      const float alpha = 1.0;
      const float beta = 0.0;

      cublasStatus_t cublas_result = cublasSgemmStridedBatched(cublas_handle,
							       op_a, op_b,
							       maxM, maxN, maxK,
							       &alpha,
							       A, lda,
							       strideA,
							       B, ldb,
							       strideB,
							       &beta,
							       C, ldc,
							       strideC,
							       batch_size);

      cudaEventRecord(end);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed, start, end);
      time += elapsed;
      assert(cublas_result == CUBLAS_STATUS_SUCCESS);
    }
    return (time / iters);
  };

  if (warmup) { runner(); }
  float time = runner();

  CUDA_CHECK(cudaFree((void*)A));
  CUDA_CHECK(cudaFree((void*)B));
  CUDA_CHECK(cudaFree((void*)C));

  return time;
}

int main(int argc, char *argv[]) {
  int batch_size = std::stoi(argv[1]);
  int num_batches = std::stoi(argv[2]);
  std::string data_file = argv[3];
  std::string mode = argv[4];
  int iters = std::stoi(argv[5]);
  int warmup = std::stoi(argv[6]);

  std::vector<int> ms;
  std::vector<int> ns;
  std::vector<int> ks;
  std::ifstream input(data_file);
  for(std::string line; getline( input, line );) {
    int m, n, k;
    input >> m >> n >> k;
    ms.push_back(m);
    ns.push_back(n);
    ks.push_back(k);
  }

  float total_time = 0;
  for (int i = 0; i < num_batches; ++i) {
    std::vector<int> bms;
    std::vector<int> bns;
    std::vector<int> bks;

    for (int j = 0; j < batch_size; ++j) {
      bms.push_back(ms[i * batch_size + j]);
      bns.push_back(ns[i * batch_size + j]);
      bks.push_back(ks[i * batch_size + j]);
    }
    total_time += testCuBLAS(batch_size, bms, bns, bks, mode, iters, warmup);
  }

  total_time /= num_batches;

  std::cout << "RESULTS," << total_time << std::endl;
}
