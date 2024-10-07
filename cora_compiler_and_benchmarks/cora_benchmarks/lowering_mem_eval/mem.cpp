#include <fstream>
#include <chrono>
#include <vector>
#include <iostream>
#include <functional>

using namespace std::chrono;

#define ceil(a, b) ((a + b - 1) / b)

double measure_time(std::function<void()> runner) {
  int w_iters = 1000;
  int a_iters = 1000;
  // int w_iters = 1;
  // int a_iters = 1;
  for (int i = 0; i < w_iters; ++i) {
    runner();
  }

  time_point<system_clock> start = system_clock::now();
  for (int i = 0; i < a_iters; ++i) {
    runner();
  }
  time_point<system_clock> end = system_clock::now();
  duration<double> exe_time = (end - start);
  return (duration_cast<microseconds>(exe_time).count() * 1.0) / a_iters;
}

int sum(std::vector<int> vec) {
  int sum = 0;
  for (auto s: vec) sum += s;
  return sum;
}

int pad_amount(std::vector<int> batch, int factor) {
  int s = sum(batch);
  if (s % factor == 0) return factor;
  else return factor - (s % factor);
}

int num_heads = 8;

class Stats {
public:
  float fusion_time = 0;
  float fusion_mem = 0;
  float tensor_time = 0;
  float tensor_mem = 0;
};

class Allocator {
public:
  virtual std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) {
    std::cout << "Unimplemented get_needed_allocs!" << std::endl;
    return {};
  }

  virtual void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) {
    std::cout << "Unimplemented construct!" << std::endl;
  }

  virtual bool is_fusion_alloc() {
    return false;
  }
};

class CSFAttnAllocator: public Allocator {
public:
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int len_sum = 0;
    for (int i = 0; i < batch_size; ++i) {
      len_sum += lens[i];
    }

    int n_idx3_ints = batch_size + 1;
    int n_idx4_ints = num_heads * len_sum + 1;
    return {n_idx3_ints, n_idx4_ints};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* idx3 = allocs[0];
    int* idx4 = allocs[1];

    int ctr3 = 0;
    int ctr4 = 0;
    int pos2 = 0;
    int pos3 = 0;

    for (int i = 0; i < batch_size; ++i) {
      idx3[pos2] = ctr3;
      for (int k = 0; k < lens[i]; ++k) {
	for (int j = 0; j < num_heads; ++j) {
	  idx4[pos3] = ctr4;
	  ctr4 += lens[i];
	  pos3++;
	}

	ctr3 += lens[i];
      }
      pos2++;
    }

    idx3[pos2] = ctr3;
    idx4[pos3] = ctr4;
  }
};

class CoRaAttnAllocator: public Allocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    return {batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* af1 = allocs[0];
    int ctr1 = 0;

    int i = 0;
    for (;i < batch_size; ++i) {
      af1[i] = ctr1;
      ctr1 += lens[i] * lens[i];
    }
    af1[i] = ctr1;
  }
};

class CSFQKTAllocator: public Allocator {
public:
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    return {batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* af1 = allocs[0];
    int ctr1 = 0;

    int i = 0;
    for (;i < batch_size; ++i) {
      af1[i] = ctr1;
      ctr1 += lens[i];
    }
    af1[i] = ctr1;
  }
};

class CoRaQKTAllocator: public Allocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    return {batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* af1 = allocs[0];
    int ctr1 = 0;

    int i = 0;
    for (;i < batch_size; ++i) {
      af1[i] = ctr1;
      ctr1 += lens[i];
    }
    af1[i] = ctr1;
  }
};

class CoRaQKTFusionAllocator: public Allocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int ctr = 0;
    for (int o = 0; o < (batch_size + 1); ++o) {
      ctr += ceil(lens[o], 64) * ceil(lens[o], 64);
    }

    return {ctr, ctr, batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* fo = allocs[0];
    int* fi = allocs[1];
    int* oif = allocs[2];

    int ctr = 0;
    for (int o = 0; o < (batch_size + 1); ++o) {
      oif[o] = ctr;
      for (int i = 0; i < ceil(lens[o], 64) * ceil(lens[o], 64); ++i) {
	fo[ctr] = o;
	fi[ctr] = i;
	ctr++;
      }
    }
  }

  bool is_fusion_alloc() override {
    return true;
  }
};

class CoRaSoftmaxFusionAllocator: public Allocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int ctr = 0;
    for (int o = 0; o < (batch_size + 1); ++o) {
      ctr += lens[o];
    }

    return {ctr, ctr, batch_size};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* fo = allocs[0];
    int* fi = allocs[1];
    int* oif = allocs[2];

    int ctr = 0;
    for (int o = 0; o < batch_size; ++o) {
      oif[o] = ctr;
      for (int i = 0; i < lens[o]; ++i) {
	fo[ctr] = o;
	fi[ctr] = i;
	ctr++;
      }
    }
  }

  bool is_fusion_alloc() override {
    return true;
  }
};

class CoRaAttnVFusionAllocator: public Allocator {
  std::vector<int> get_needed_allocs(int batch_size, std::vector<int> lens) override {
    int ctr = 0;
    for (int o = 0; o < (batch_size + 1); ++o) {
      ctr += ceil(lens[o], 64);
    }

    return {ctr, ctr, batch_size + 1};
  }

  void construct(std::vector<int*> allocs, int batch_size, std::vector<int> lens) override {
    int* fo = allocs[0];
    int* fi = allocs[1];
    int* oif = allocs[2];

    int ctr = 0;
    for (int o = 0; o < (batch_size + 1); ++o) {
      oif[o] = ctr;
      for (int i = 0; i < ceil(lens[o], 64); ++i) {
	fo[ctr] = o;
	fi[ctr] = i;
	ctr++;
      }
    }
  }

  bool is_fusion_alloc() override {
    return true;
  }
};

std::vector<int*> alloc_memory(std::vector<int> sizes) {
  std::vector<int*> ret;
  for (auto size: sizes) {
    ret.push_back(new int[size]);
  }
  return ret;
}

void free_allocs(std::vector<int*> allocs) {
  for (auto alloc: allocs) {
    delete[] alloc;
  }
}

Stats run(int batch_size, std::vector<int> lens, std::vector<Allocator*> allocators) {
  float fusion_time = 0;
  float fusion_mem = 0;
  float tensor_time = 0;
  float tensor_mem = 0;

  for (Allocator* allocator: allocators) {
    std::vector<int> allocs_needed = allocator->get_needed_allocs(batch_size, lens);
    std::vector<int*> allocated_mem = alloc_memory(allocs_needed);

    auto runner = [&] {
      allocator->construct(allocated_mem, batch_size, lens);
    };

    double time = measure_time(runner);

    if (allocator->is_fusion_alloc()) {
      fusion_time += time;
      fusion_mem += sum(allocs_needed);
    } else {
      tensor_time += time;
      tensor_mem += sum(allocs_needed);
    }

    free_allocs(allocated_mem);
  }

  return {fusion_time, fusion_mem, tensor_time, tensor_mem};
}

int main(int argc, char** argv) {
  int batch_size = std::stoi(argv[1]);
  int num_batches = std::stoi(argv[2]);
  std::string data_file = argv[3];
  std::string mode = argv[4];

  std::fstream fs;
  fs.open(data_file);
  if (!fs.is_open()){
    printf("Error opening input\n");
    exit(EXIT_FAILURE);
  }

  std::vector<Allocator*> allocators;
  if (mode == "csf") {
    allocators.push_back(new CSFAttnAllocator());
    allocators.push_back(new CSFQKTAllocator());
  } else {
    allocators.push_back(new CoRaAttnAllocator());
    allocators.push_back(new CoRaQKTAllocator());
    allocators.push_back(new CoRaQKTFusionAllocator());
    allocators.push_back(new CoRaSoftmaxFusionAllocator());
    allocators.push_back(new CoRaAttnVFusionAllocator());
  }

  Stats aggregate;
  for (int i = 0; i < num_batches; ++i) {
    std::vector<int> lens(batch_size, -1);

    for (int j = 0; j < batch_size; ++j){
      fs >> lens[j];
    }

    if (mode == "cora") {
      lens.push_back(pad_amount(lens, 64));
    }

    Stats stats = run(batch_size, lens, allocators);

    aggregate.fusion_time += stats.fusion_time;
    aggregate.fusion_mem += stats.fusion_mem;
    aggregate.tensor_time += stats.tensor_time;
    aggregate.tensor_mem += stats.tensor_mem;
  }

  aggregate.fusion_time /= num_batches;
  aggregate.fusion_mem /= num_batches;
  aggregate.tensor_time /= num_batches;
  aggregate.tensor_mem /= num_batches;

  std::cout << "RESULTS," << aggregate.fusion_time << "," << aggregate.tensor_time << std::endl;
  std::cout << "MEM," << aggregate.fusion_mem << "," << aggregate.tensor_mem << std::endl;
  return 0;
}
