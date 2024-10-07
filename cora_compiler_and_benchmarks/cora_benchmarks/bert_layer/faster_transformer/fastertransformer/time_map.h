#pragma once

#define OP_TIMES 1

#define START_OPTIME_MEASUREMENT  \
  {cudaEventCreate(&start);	  \
   cudaEventCreate(&end);            \
   cudaEventRecord(start);}	     \

#define END_OPTIME_MEASUREMENT(op)	      	   \
  {cudaEventRecord(end);			   \
   cudaEventSynchronize(end);			   \
   cudaEventElapsedTime(&elapsed, start, end);	     \
   TimeMap::AddTime(op, elapsed);}		     \


#include <unordered_map>
#include <iostream>

namespace fastertransformer {
  class TimeMap {
  public:
    static void AddTime(std::string op, float time) {
      if (do_profile) {
	// std::cout << "RECORD " << op << " " << time << std::endl;
	auto it = times.find(op);
	if (it == times.end()) {
	  times[op] = time;
	} else {
	  it->second += time;
	}
      }
    }

    static void Print() {
      float sum = 0.0;
      for (auto it: times) {
	std::cout << "RESULTS," << it.first << "," << it.second << std::endl;
	sum += it.second;
      }
      std::cout << "SUM," << sum << std::endl;
    }

    static void StopProfiling() {
      do_profile = false;
    }

    static void StartProfiling() {
      do_profile = true;
    }

    static void MeanBy(int ite) {
      for (auto& it: times) {
	// times[it.first] /= ite;
	it.second /= ite;
      }
    }

  private:
    static std::unordered_map<std::string, float> times;
    static bool do_profile;
  };
}
