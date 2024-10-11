#include <cublas_v2.h>
#include <cuda.h>
#include <random>
#include <fstream>

const char* cublasGetErrorString(cublasStatus_t status) {
  switch(status) {
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
  return "unknown error";
}

#define CUDA_CHECK(status)                                              \
  {                                                                     \
    cudaError_t error = status;                                         \
    if (error != cudaSuccess) {                                         \
      std::cerr << "Got bad cuda status: " << cudaGetErrorString(error) \
                << " at line: " << __LINE__ << std::endl;               \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  }

const int num_heads = 8;
const int head_size = 64;

std::random_device rd;
// std::mt19937 rng(rd());
std::mt19937 rng(0);

std::pair<int, int> blockize(const std::vector<int>& lens, std::vector<int>& blocked, int block_size) {
  int num_sq_blocks = 0;
  int num_len_blocks = 0;
  for (int i = 0; i < lens.size(); ++i) {
    int b = static_cast<int>(ceil(lens[i] * 1.0 / block_size));
    blocked[i] = b;
    num_sq_blocks += b * b;
    num_len_blocks += b;
  }
  return std::make_pair(num_sq_blocks, num_len_blocks);
}

void create_random_lens(int max_seq_len, int avg_seq_len, int num, std::vector<int>& lens) {
  int min = 2 * avg_seq_len - max_seq_len;
  int max = max_seq_len;
  std::uniform_int_distribution<int> uni(min, max);
  lens.reserve(num);
  for (int i = 0; i < num; ++i) {
    lens[i] = uni(rng);
  }
}

void read_lengths(std::string filepath, std::vector<int>& lengths) {
  std::ifstream file(filepath);
  std::string str;
  while (std::getline(file, str)) {
    lengths.push_back(std::stoi(str));
  }
}


#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define CEIL_MULTIPLE(a, b) ((b) * (((a) + (b) - 1) / (b)))
