#include <chrono>
#include <functional>
#include <assert.h>
#include <algorithm>
#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <math.h>
#include <mkl.h>

int find_max(std::vector<int> v) {
  int max = -10000;
  for (auto e: v) {
    if (e > max) max = e;
  }
  return max;
}

float measure_time(std::function<float()> runner, int w_iters, int a_iters) {
  for (int i = 0; i < w_iters; ++i) {
    runner();
  }

  float exe_time = 0.0;
  for (int i = 0; i < a_iters; ++i) {
    exe_time += runner();
  }

  return exe_time / a_iters;
}

float testUBatch(int batch_size, std::vector<int> ms, std::vector<int> ns, std::vector<int> ks, int iters, int warmup) {
  int maxM = find_max(ms);
  int maxN = find_max(ns);
  int maxK = find_max(ks);

  // int maxM = 1408;
  // int maxN = 1408;
  // int maxK = 1408;

  long a_size = maxM * maxK;
  long b_size = maxK * maxN;
  long c_size = maxM * maxN;

  const float* a = static_cast<const float*>(malloc(batch_size * a_size * sizeof(float)));
  const float* b = static_cast<const float*>(malloc(batch_size * b_size * sizeof(float)));
  float* c = static_cast<float*>(malloc(batch_size * c_size * sizeof(float)));


  auto runner = [&] {
    using namespace std::chrono;
    time_point<system_clock> start = system_clock::now();
    cblas_sgemm_batch_strided(CblasRowMajor,
			      CblasNoTrans, CblasNoTrans,
			      maxM, maxN, maxK,
			      1.0,
			      a, maxK, a_size,
			      b, maxN, b_size,
			      0.0,
			      c, maxN, c_size,
			      batch_size);
    time_point<system_clock> end = system_clock::now();
    std::chrono::duration<float> exe_time = (end - start);
    return duration_cast<microseconds>(exe_time).count();
  };

  return measure_time(runner, warmup ? iters : 0, iters);
}

float testVBatch(int batch_size, std::vector<int> ms, std::vector<int> ns, std::vector<int> ks, int iters, int warmup) {
  int maxM = find_max(ms);
  int maxN = find_max(ns);
  int maxK = find_max(ks);

  int group_count = batch_size;
  int total_batch_count = batch_size;
  CBLAS_TRANSPOSE* transa_array = new CBLAS_TRANSPOSE[group_count];
  CBLAS_TRANSPOSE* transb_array = new CBLAS_TRANSPOSE[group_count];

  float* alpha_array = new float[group_count];
  float* beta_array = new float[group_count];

  int* m_array = new int[group_count];
  int* n_array = new int[group_count];
  int* k_array = new int[group_count];

  int* lda_array = new int[group_count];
  int* ldb_array = new int[group_count];
  int* ldc_array = new int[group_count];

  int* group_size_array = new int[group_count];

  float** a_array = new float*[total_batch_count];
  float** b_array = new float*[total_batch_count];
  float** c_array = new float*[total_batch_count];

  long a_size = maxM * maxK;
  long b_size = maxK * maxN;
  long c_size = maxM * maxN;

  float* a_start = static_cast<float*>(malloc(batch_size * a_size * sizeof(float)));
  float* b_start = static_cast<float*>(malloc(batch_size * b_size * sizeof(float)));
  float* c_start = static_cast<float*>(malloc(batch_size * c_size * sizeof(float)));

  int idx = 0;
  for (int i = 0; i < group_count; ++i) {
    transa_array[i] = CblasNoTrans;
    transb_array[i] = CblasNoTrans;

    alpha_array[i] = 1.0;
    beta_array[i] = 0.0;

    m_array[i] = ms[i];
    n_array[i] = ns[i];
    k_array[i] = ks[i];

    lda_array[i] = ks[i];
    ldb_array[i] = ns[i];
    ldc_array[i] = ns[i];

    a_array[i] = a_start + i * a_size;
    b_array[i] = b_start + i * b_size;
    c_array[i] = c_start + i * c_size;

    group_size_array[i] = 1;
  }

  auto runner = [&] {
    using namespace std::chrono;
    time_point<system_clock> start = system_clock::now();
    cblas_sgemm_batch(CblasRowMajor,
		      transa_array, transb_array,
		      m_array, n_array, k_array,
		      alpha_array,
		      const_cast<const float**>(a_array), lda_array,
		      const_cast<const float**>(b_array), ldb_array,
		      beta_array,
		      c_array, ldc_array,
		      group_count, group_size_array);
    time_point<system_clock> end = system_clock::now();
    std::chrono::duration<float> exe_time = (end - start);
    return duration_cast<microseconds>(exe_time).count();
  };

  return measure_time(runner, warmup ? iters : 0, iters);
}

int main(int argc, char *argv[]) {
  mkl_set_threading_layer(MKL_THREADING_GNU);
  int batch_size = std::stoi(argv[1]);
  int num_batches = std::stoi(argv[2]);
  int add_pad = std::stoi(argv[3]);
  std::string data_file = argv[4];
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

  float time = 0;
  for (int i = 0; i < num_batches; ++i) {
    std::vector<int> bms;
    std::vector<int> bns;
    std::vector<int> bks;

    for (int j = 0; j < batch_size; ++j) {
      bms.push_back(ms[i * batch_size + j]);
      bns.push_back(ns[i * batch_size + j]);
      bks.push_back(ks[i * batch_size + j]);
    }
    if (add_pad) {
      time += testUBatch(batch_size, bms, bns, bks, iters, warmup);
    }  else {
      time += testVBatch(batch_size, bms, bns, bks, iters, warmup);
    }
  }

  time /= num_batches;
  time /= 1000;

  std::cout << "RESULTS," << time << std::endl;
}
