#include <stdio.h>
#include <stdlib.h>

#define TIME_MAX 5.0
#define TRIAL_MAX 10000

int dummySize = 3000000;
double* dummyA = NULL;
double* dummyB = NULL;

__attribute__((noinline)) 
double clear_cache() {
  double ret = 0.0;
  if (!dummyA) {
    dummyA = (double*)(malloc(dummySize*sizeof(double)));
    dummyB = (double*)(malloc(dummySize*sizeof(double)));
  }
  for (int i=0; i< 100; i++) {
    dummyA[rand() % dummySize] = rand()/RAND_MAX;
    dummyB[rand() % dummySize] = rand()/RAND_MAX;
  }
  for (int i=0; i<dummySize; i++) {
    ret += dummyA[i] * dummyB[i];
  }
  return ret;
}

template <typename Setup, typename Test>
long long benchmark(Setup setup, Test test, bool cold_cache=true){
  auto time_total = std::chrono::high_resolution_clock::duration(0);
  auto time_min = std::chrono::high_resolution_clock::duration(0);
  int trial = 0;
  while(trial < TRIAL_MAX){
    setup();
    if (cold_cache) clear_cache();
    auto tic = std::chrono::high_resolution_clock::now();
    test();
    auto toc = std::chrono::high_resolution_clock::now();
    if(toc < tic){
      exit(EXIT_FAILURE);
    }
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(toc-tic);
    trial++;
    if(trial == 1 || time < time_min){
      time_min = time;
    }
    time_total += time;
    if(time_total.count() * 1e-9 > TIME_MAX){
      break;
    }
  }
  return (long long) time_min.count();
}
