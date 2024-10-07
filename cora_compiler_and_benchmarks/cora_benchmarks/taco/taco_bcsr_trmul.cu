#include <iostream>
#include "taco.h"
#include "utils.cuh"

using namespace taco;
using namespace std::chrono;

const IndexVar io("io"), jo("jo"), ii("ii"), ji("ji");
int WARP_SIZE = 32;

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<float> A, int m) {
  const IndexVar fi("fi");
  return stmt.reorder({io, jo, ii, ji})
    .parallelize(io, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
    .fuse(ii, ji, fi)
    .parallelize(fi, ParallelUnit::GPUThread, OutputRaceStrategy::IgnoreRaces);
}

__global__ void computeDeviceKernel0(taco_tensor_t * __restrict__ A,
				     taco_tensor_t * __restrict__ B,
				     taco_tensor_t * __restrict__ C) {
  int A3_dimension = (int)(A->dimensions[2]);
  int A4_dimension = (int)(A->dimensions[3]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B3_dimension = (int)(B->dimensions[2]);
  int B4_dimension = (int)(B->dimensions[3]);
  int* __restrict__ B2_pos = (int*)(B->indices[1][0]);
  int* __restrict__ B2_crd = (int*)(B->indices[1][1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  int C3_dimension = (int)(C->dimensions[2]);
  int C4_dimension = (int)(C->dimensions[3]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t io = blockIdx.x;
  int32_t fi = (threadIdx.x % (B3_dimension * B4_dimension));
  if (threadIdx.x >= (B3_dimension * B4_dimension)) {
    return;
  }

  int32_t joA = A2_pos[io];
  int32_t pA2_end = A2_pos[(io + 1)];
  int32_t joB = B2_pos[io];
  int32_t pB2_end = B2_pos[(io + 1)];

  while (joA < pA2_end && joB < pB2_end) {
    int32_t joA0 = A2_crd[joA];
    int32_t joB0 = B2_crd[joB];
    int32_t jo = TACO_MIN(joA0,joB0);
    if (joA0 == jo && joB0 == jo) {
      int32_t joC = io * C1_dimension + jo;
      int32_t ii = fi / B4_dimension;
      int32_t iiB = joB * B3_dimension + ii;
      int32_t iiA = joA * A3_dimension + ii;
      int32_t iiC = joC * C3_dimension + ii;
      if (ii >= B3_dimension)
        return;

      int32_t ji = fi % B4_dimension;
      int32_t jiB = iiB * B4_dimension + ji;
      int32_t jiA = iiA * A4_dimension + ji;
      int32_t jiC = iiC * C4_dimension + ji;
      if (ji >= B4_dimension)
        return;

      C_vals[jiC] = A_vals[jiA] * B_vals[jiB];
    }
    joA = joA + (int32_t)(joA0 == jo);
    joB = joB + (int32_t)(joB0 == jo);
  }
}

float compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B, int m, int iters) {
  int B1_dimension = (int)(B->dimensions[0]);
  int B3_dimension = (int)(B->dimensions[2]);
  int B4_dimension = (int)(B->dimensions[3]);

  gpuErrchk(cudaMallocManaged((void**)&(C->vals), sizeof(float) * m * m));

  int num_blocks = B1_dimension;
  int num_threads = B3_dimension * B4_dimension;
  std::cout << "NB/NT " << num_blocks << " " << num_threads << std::endl;

  cudaEvent_t start, end;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  for (int i = 0; i < iters; ++i) {
    computeDeviceKernel0<<<num_blocks, num_threads>>>(A, B, C);
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
  int bs = std::atoi(argv[2]);
  int mb = m/bs;
  std::cout << "M/BS/Mb " << m << " " << bs << " " << mb << std::endl;
  Tensor<float> A("A", {mb, mb, bs, bs}, {Dense, Compressed, Dense, Dense});
  Tensor<float> B("B", {mb, mb, bs, bs}, {Dense, Compressed, Dense, Dense});
  Tensor<float> C("C", {mb, mb, bs, bs}, Format({{Dense, Dense, Dense, Dense}}));

  for (int i = 0; i < mb; ++i) {
    for (int j = 0; j < i + 1; ++j) {
      for (int ii = 0; ii < bs; ++ii) {
  	for (int ji = 0; ji < bs; ++ji) {
  	  float rand_float = (float)rand()/(float)(RAND_MAX);
  	  A.insert({i, j, ii, ji}, rand_float);
  	  B.insert({i, j, ii, ji}, rand_float);
  	}
      }
    }
    for (int j = i + 1; j < mb; j++) {
      for (int ii = 0; ii < bs; ++ii) {
  	for (int ji = 0; ji < bs; ++ji) {
  	  float rand_float = (float)rand()/(float)(RAND_MAX);
  	  C.insert({i, j, ii, ji}, rand_float);
  	}
      }
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

  // C(io, jo, ii, ji) = A(io, jo, ii, ji) * B(io, jo, ii, ji);

  // IndexStmt stmt = C.getAssignment().concretize();
  // stmt = scheduleSpMMGPU(stmt, A, m);
  // std::cout << stmt << std::endl;
  // C.compile(stmt);

  // std::cout << C.getSource() << std::endl;
}
