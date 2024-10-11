#include <iostream>
#include "taco.h"
#include "utils.cuh"

using namespace taco;
using namespace std::chrono;

const IndexVar io("io"), jo("jo"), ko("ko"), ii("ii"), ji("ji"), ki("ki");
int WARP_SIZE = 32;

IndexStmt scheduleSpMMGPU(IndexStmt stmt, Tensor<float> A, int m, int bs, IndexExpr precomputedAExpr,
			  int NNZ_PER_WARP=1, int BLOCK_SIZE=256) {
  int NNZ_PER_TB = NNZ_PER_WARP * (BLOCK_SIZE / WARP_SIZE);
  IndexVar f("f"), fpos("fpos"), block("block"), fpos1("fpos1"), warp("warp"), nnz("nnz"), nnz_pre("nnz_pre");
  IndexVar dense_val_unbounded("dense_val_unbounded"), dense_val("dense_val"), thread("thread");
  IndexVar thread_nz("thread_nz");
  TensorVar precomputedA("precomputedA", Type(Float32, {Dimension(nnz)}), taco::dense);
  return stmt.reorder({io, jo, ko, ii, ji, ki})
  .fuse(io, jo, f)
    .pos(f, fpos, A(io, jo, ii, ji))
    .split(fpos, block, fpos1, NNZ_PER_TB)
    .split(fpos1, warp, nnz, NNZ_PER_WARP)
    .split(ko, dense_val_unbounded, thread, WARP_SIZE)
    .reorder({block, warp, thread, dense_val_unbounded, nnz})
    .bound(dense_val_unbounded, dense_val, (m / bs) / WARP_SIZE, BoundType::MaxExact)
    .unroll(dense_val, 4)
    .parallelize(block, ParallelUnit::GPUBlock, OutputRaceStrategy::IgnoreRaces)
    .parallelize(warp, ParallelUnit::GPUWarp, OutputRaceStrategy::IgnoreRaces)
    .parallelize(thread, ParallelUnit::GPUThread, OutputRaceStrategy::Atomics);
}

template<int BLOCK_SIZE, int TBLOCK_SIZE, int WARP_SIZE>
__global__ void computeKernel(taco_tensor_t * __restrict__ A,
			      taco_tensor_t * __restrict__ B,
			      taco_tensor_t * __restrict__ C,
			      int32_t* io_blockStarts, int mb,
			      float alpha) {
  int A1_dimension = (int)(A->dimensions[0]);
  int A3_dimension = (int)(A->dimensions[2]);
  int A4_dimension = (int)(A->dimensions[3]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  int B3_dimension = (int)(B->dimensions[2]);
  int B4_dimension = (int)(B->dimensions[3]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C2_dimension = (int)(C->dimensions[1]);
  int C3_dimension = (int)(C->dimensions[2]);
  int C4_dimension = (int)(C->dimensions[3]);
  float* __restrict__ C_vals = (float*)(C->vals);
  const int NUM_WARPS = TBLOCK_SIZE / WARP_SIZE;

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (WARP_SIZE));
  int32_t warp = (threadIdx.x / WARP_SIZE);
  if (threadIdx.x >= TBLOCK_SIZE) {
    return;
  }

  float Cl[BLOCK_SIZE];

  #pragma unroll 4
  for (int32_t dense_val = 0; dense_val < (mb + WARP_SIZE - 1)/WARP_SIZE; dense_val++) {
    int32_t ko = dense_val * WARP_SIZE + thread;
    // if (ko >= mb) break;

    int32_t pA2_begin = io_blockStarts[block];
    int32_t pA2_end = io_blockStarts[(block + 1)];
    int32_t fposA = block * NUM_WARPS + warp;
    int32_t io_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t io = io_pos;
    for (int32_t nnz = 0; nnz < 1; nnz++) {
      int32_t fposA = block * NUM_WARPS + warp;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      int32_t koB = f * B2_dimension + ko;
      while (fposA == A2_pos[(io_pos + 1)]) {
        io_pos = io_pos + 1;
        io = io_pos;
      }
      int32_t koC = io * C2_dimension + ko;
      for (int32_t ii = 0; ii < A3_dimension; ii++) {
	for (int i = 0; i < BLOCK_SIZE; ++i) {
	  Cl[i] = 0;
	}
        int32_t iiC = koC * C3_dimension + ii;
        int32_t iiA = fposA * A3_dimension + ii;
        for (int32_t ji = 0; ji < A4_dimension; ji++) {
          int32_t jiB = koB * B3_dimension + ji;
          int32_t jiA = iiA * A4_dimension + ji;
          for (int32_t ki = 0; ki < B4_dimension; ki++) {
            int32_t kiC = iiC * C4_dimension + ki;
            int32_t kiB = jiB * B4_dimension + ki;
	    float av = A_vals[jiA];
	    float bv = B_vals[kiB];
	    Cl[ji] += av*bv;
          }
        }
	for (int32_t ki = 0; ki < B4_dimension; ki++) {
	  int32_t kiC = iiC * C4_dimension + ki;
	  // C_vals[kiC] = alpha*Cl[ki];
	  atomicAdd(&(C_vals[kiC]), alpha*Cl[ki]);
	}
      }
    }
  }
}

float compute(taco_tensor_t *C, taco_tensor_t *B, taco_tensor_t *A, int m, int bs, float alpha, int iters) {
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);

  int num_bcsr_blocks = A2_pos[A1_dimension];
  int32_t* io_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&io_blockStarts, sizeof(int32_t) * 5000));
  gpuErrchk(cudaMallocManaged((void**)&(C->vals), sizeof(float) * m * m));

  cudaEvent_t start, end;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  if (m == 128) {
    const int warp_size = 4;
    const int block_size = 8;
    int num_tblocks = num_bcsr_blocks / (block_size / warp_size);
    for (int i = 0; i < iters; ++i) {
      cudaMemsetAsync(C->vals, 0, sizeof(float) * m * m);
      io_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, io_blockStarts, 0, A1_dimension,
							  block_size / warp_size, block_size, num_tblocks);
      computeKernel<32, block_size, warp_size><<<num_tblocks, block_size>>>
	(A, B, C, io_blockStarts, m/bs, alpha);
    }
  } else if (m == 512) {
    const int warp_size = 16;
    const int block_size = 32;
    int num_tblocks = num_bcsr_blocks / (block_size / warp_size);
    for (int i = 0; i < iters; ++i) {
      cudaMemsetAsync(C->vals, 0, sizeof(float) * m * m);
      io_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, io_blockStarts, 0, A1_dimension,
							  block_size / warp_size, block_size, num_tblocks);
      computeKernel<32, block_size, warp_size><<<num_tblocks, block_size>>>
	(A, B, C, io_blockStarts, m/bs, alpha);
    }
  } else {
    const int warp_size = 32;
    const int block_size = 256;
    int num_tblocks = num_bcsr_blocks / (block_size / warp_size);
    for (int i = 0; i < iters; ++i) {
      cudaMemsetAsync(C->vals, 0, sizeof(float) * m * m);
      io_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, io_blockStarts, 0, A1_dimension,
							  block_size / warp_size, block_size, num_tblocks);
      computeKernel<32, block_size, warp_size><<<num_tblocks, block_size>>>
	(A, B, C, io_blockStarts, m/bs, alpha);
    }
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, start, end);
  gpuErrchk(cudaPeekAtLastError());
  gpuErrchk(cudaDeviceSynchronize());

  cudaFree(io_blockStarts);
  return elapsed;
}

int main(int argc, char* argv[]) {
  int m = std::atoi(argv[1]);
  int bs = std::atoi(argv[2]);
  int mb = m/bs;
  Tensor<float> A("A", {mb, mb, bs, bs}, {Dense, Compressed, Dense, Dense});
  Tensor<float> B("B", {mb, mb, bs, bs}, {Dense, Dense, Dense, Dense});
  Tensor<float> C("C", {mb, mb, bs, bs}, {Dense, Dense, Dense, Dense});

  for (int i = 0; i < mb; ++i) {
    for (int j = 0; j < i + 1; ++j) {
      for (int ii = 0; ii < bs; ++ii) {
  	for (int ji = 0; ji < bs; ++ji) {
  	  float rand_float = 0.1;//(float)rand()/(float)(RAND_MAX);
  	  A.insert({i, j, ii, ji}, rand_float);
  	  B.insert({i, j, ii, ji}, rand_float);
  	  C.insert({i, j, ii, ji}, rand_float);
  	}
      }
    }
    for (int j = i + 1; j < mb; j++) {
      for (int ii = 0; ii < bs; ++ii) {
  	for (int ji = 0; ji < bs; ++ji) {
  	  float rand_float = 0.1;//(float)rand()/(float)(RAND_MAX);
  	  B.insert({i, j, ii, ji}, rand_float);
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
  float alpha = 0.9;
  // Warm up
  compute(Ct, Bt, At, m, bs, alpha, witers);

  float time = compute(Ct, Bt, At, m, bs, alpha, iters);
  time /= iters;
  std::cout << "RESULTS," << time << std::endl;

  // for (int i = 0; i < m; ++i) {
    // for (int j = 0; j < m; ++j) {
      // std::cout << ((float*)Ct->vals)[i*m+j] << " ";
    // }
    // std::cout << std::endl;
  // }

  // IndexExpr precomputedA = A(io, jo, ii, ji);
  // IndexExpr precomputedB = B(jo, ko, ji, ki);
  // C(io, ko, ii, ki) += precomputedB * precomputedA;

  // IndexStmt stmt = C.getAssignment().concretize();
  // stmt = scheduleSpMMGPU(stmt, A, m, bs, precomputedA);

  // C.compile(stmt);
}
