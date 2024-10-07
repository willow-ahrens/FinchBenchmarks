#include <iostream>
#include "taco.h"
#include "utils.cuh"

using namespace taco;
using namespace std::chrono;

__global__ void computeDeviceKernel0(taco_tensor_t * __restrict__ A,
				     taco_tensor_t * __restrict__ B,
				     taco_tensor_t * __restrict__ C) {
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B1_dimension = (int)(B->dimensions[0]);
  int* __restrict__ B2_pos = (int*)(B->indices[1][0]);
  int* __restrict__ B2_crd = (int*)(B->indices[1][1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C2_dimension = (int)(C->dimensions[1]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t i0 = blockIdx.x;
  int32_t i1 = threadIdx.x;
  if (threadIdx.x >= 64) {
    return;
  }

  int32_t i = i0 * 64 + i1;
  if (i >= B1_dimension)
    return;

  int32_t jA = A2_pos[i];
  int32_t pA2_end = A2_pos[(i + 1)];
  int32_t jB = B2_pos[i];
  int32_t pB2_end = B2_pos[(i + 1)];

  while (jA < pA2_end && jB < pB2_end) {
    int32_t jA0 = A2_crd[jA];
    int32_t jB0 = B2_crd[jB];
    int32_t j = TACO_MIN(jA0,jB0);
    if (jA0 == j && jB0 == j) {
      int32_t jC = i * C2_dimension + j;
      C_vals[jC] = A_vals[jA] + B_vals[jB];
    }
    else if (jA0 == j) {
      int32_t jC = i * C2_dimension + j;
      C_vals[jC] = A_vals[jA];
    }
    else {
      int32_t jC = i * C2_dimension + j;
      C_vals[jC] = B_vals[jB];
    }
    jA = jA + (int32_t)(jA0 == j);
    jB = jB + (int32_t)(jB0 == j);
  }
  while (jA < pA2_end) {
    int32_t j = A2_crd[jA];
    int32_t jC = i * C2_dimension + j;
    C_vals[jC] = A_vals[jA];
    jA = jA + 1;
  }
  while (jB < pB2_end) {
    int32_t j = B2_crd[jB];
    int32_t jC = i * C2_dimension + j;
    C_vals[jC] = B_vals[jB];
    jB = jB + 1;
  }
}

float compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B, int m, int iters) {
  int B1_dimension = (int)(B->dimensions[0]);
  gpuErrchk(cudaMallocManaged((void**)&(C->vals), sizeof(float) * m * m));

  cudaEvent_t start, end;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  for (int i = 0; i < iters; ++i) {
    computeDeviceKernel0<<<(B1_dimension + 63) / 64, 64>>>(A, B, C);
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());
  return elapsed;
}

int main(int argc, char* argv[]) {
  int m = std::atoi(argv[1]);
  Tensor<float> A("A", {m, m}, CSR);
  Tensor<float> B("B", {m, m}, CSR);
  Tensor<float> C("C", {m, m}, Format({{Dense, Dense}, {1, 0}}));

  srand(434321);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < i + 1; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      // float rand_float = 0.1;
      B.insert({i, j}, rand_float);
      C.insert({i, j}, rand_float);
      A.insert({i, j}, rand_float);
    }
    for (int j = i + 1; j < m; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      // float rand_float = 0.1;
      C.insert({i, j}, rand_float);
    }
  }
  A.pack();
  B.pack();

  auto At = A.getTacoTensorT();
  auto Bt = B.getTacoTensorT();
  auto Ct = C.getTacoTensorT();

  int witers = 100;
  int iters = 100;
  // Warm up
  compute(Ct, At, Bt, m, witers);

  float time = compute(Ct, At, Bt, m, iters);
  time /= iters;

  // float* vals = (float*)Ct->vals;
  // for (int i = 0; i < m; ++i) {
  //   for (int j = 0; j < m; j++) {
  //     std::cout << vals[i*m + j] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  std::cout << "RESULTS," << time << std::endl;
}
