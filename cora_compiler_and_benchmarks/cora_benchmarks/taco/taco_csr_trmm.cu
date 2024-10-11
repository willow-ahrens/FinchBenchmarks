#include <iostream>
#include "taco.h"
#include "utils.cuh"

using namespace taco;
using namespace std::chrono;

const IndexVar i("i"), j("j"), k("k"), l("l"), m("m"), n("n");
int WARP_SIZE = 32;

__global__ void computeCSRLB(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B,
			     taco_tensor_t * __restrict__ C, int32_t* i_blockStarts, int32_t m, float alpha) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 256) {
    return;
  }

  __shared__ float precomputedA_ALL[256];
  float * precomputedA = precomputedA_ALL + warp * 32;
  for (int32_t pprecomputedA = 0; pprecomputedA < 8; pprecomputedA++) {
    precomputedA[pprecomputedA] = 0.0;
  }
  for (int32_t nnz = 0; nnz < 8; nnz++) {
    int32_t fpos1 = warp * 8 + nnz;
    int32_t fposA = block * 64 + fpos1;
    if (fposA >= A2_pos[A1_dimension])
      break;

    precomputedA[nnz] = precomputedA[nnz] + A_vals[fposA];
  }
  #pragma unroll 4
  for (int32_t dense_val = 0; dense_val < m / 32; dense_val++) {
    int32_t k = dense_val * 32 + thread;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 8;
    int32_t fposA = block * 64 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 8; nnz++) {
      int32_t fpos1 = warp * 8 + nnz;
      int32_t fposA = block * 64 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      int32_t kB = f * B2_dimension + k;
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = k * C1_dimension + i;
      atomicAdd(&C_vals[iC], alpha*B_vals[kB] * precomputedA[nnz]);
    }
  }
}

__global__ void computeCSRNoLB(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B,
			       taco_tensor_t * __restrict__ C, int32_t* i_blockStarts, int32_t m,
			       float alpha) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C2_dimension = (int)(C->dimensions[1]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t f1 = blockIdx.x;
  int32_t i0 = f1 / ((B2_dimension + 63) / 64);
  int32_t j0 = f1 % ((B2_dimension + 63) / 64);
  int32_t f2 = threadIdx.x;
  int32_t i10 = f2 / 8;
  int32_t j10 = f2 % 8;
  for (int32_t i11 = 0; i11 < 8; i11++) {
    int32_t i1 = i10 * 8 + i11;
    int32_t i = i0 * 64 + i1;
    if (i >= A1_dimension)
      continue;

    for (int32_t kA = A2_pos[i]; kA < A2_pos[(i + 1)]; kA++) {
      int32_t k = A2_crd[kA];
      for (int32_t j11 = 0; j11 < 8; j11++) {
	int32_t j1 = j10 * 8 + j11;
	int32_t j = j0 * 64 + j1;
	int32_t jB = k * B2_dimension + j;
	int32_t jC = i * C2_dimension + j;
	if (j >= B2_dimension)
	  continue;

	C_vals[jC] = C_vals[jC] + alpha*A_vals[kA] * B_vals[jB];
      }
    }
  }
}

float compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B, int32_t m, float alpha, int32_t mode, int32_t iters) {
  int A1_dimension = (int)(A->dimensions[0]);
  int B2_dimension = (int)(B->dimensions[1]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);

  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 63) / 64 + 1)));
  gpuErrchk(cudaMallocManaged((void**)&(C->vals), sizeof(float) * m * m));

  cudaEvent_t start, end;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  if (mode == 0) {
    for (int i = 0; i < iters; ++i) {
      cudaMemsetAsync(C->vals, 0, sizeof(float) * m * m);
      i_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, (int32_t) 0, A1_dimension,
							 (int32_t) 64, (int32_t) 256, ((A2_pos[A1_dimension] + 63) / 64));
      computeCSRLB<<<(A2_pos[A1_dimension] + 63) / 64, (32 * 8)>>>(A, B, C, i_blockStarts, m, alpha);
    }
  } else {
    int num_blocks = (((A1_dimension + 63) / 64) * ((B2_dimension + 63) / 64));
    for (int i = 0; i < iters; ++i) {
      cudaMemsetAsync(C->vals, 0, sizeof(float) * m * m);
      computeCSRNoLB<<<num_blocks, (32 * 8)>>>(A, B, C, i_blockStarts, m, alpha);
    }
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(i_blockStarts);
  return elapsed;
}

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<float> A, int m, IndexExpr precomputedExprA,
			  IndexExpr precomputedExprB, int NNZ_PER_WARP=8, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), nnz_pre("nnz_pre");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  TensorVar precomputedA("precomputedA", Type(Float32, {Dimension(nnz)}), taco::dense);
  TensorVar precomputedB("precomputedB", Type(Float32, {Dimension(nnz)}), taco::dense);
  return stmt.reorder({i, j, k})
          .fuse(i, j, f)
          .pos(f, fpos, A(i, j))
          .split(fpos, block, fpos1, NNZ_PER_TB)
          .split(fpos1, warp, nnz, NNZ_PER_WARP)
          .split(k, dense_val_unbounded, thread, WARP_SIZE)
          .reorder({block, warp, thread, dense_val_unbounded, nnz})
          .precompute(precomputedExprA, nnz, nnz, precomputedA)
    .bound(dense_val_unbounded, dense_val, m/WARP_SIZE, BoundType::MaxExact)
          .unroll(dense_val, 4)
          .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
          .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
          .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

int main(int argc, char* argv[]) {
  int m = std::atoi(argv[1]);
  int mode = 0;//std::atoi(argv[2]);
  int NUM_I = m;
  int NUM_J = m;
  int NUM_K = m;
  Tensor<float> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<float> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));

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
      B.insert({i, j}, rand_float);
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
  float alpha = 0.03;
  // Warm up
  compute(Ct, At, Bt, m, alpha, mode, witers);

  float time = compute(Ct, At, Bt, m, alpha, mode, iters);
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
