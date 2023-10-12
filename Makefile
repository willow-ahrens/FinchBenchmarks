CC = gcc
CXX = g++
LD = ld
CXXFLAGS += -std=c++11
LDLIBS +=

ifeq ("$(shell uname)","Darwin")
export NPROC_VAL := $(shell sysctl -n hw.logicalcpu_max )
else
export NPROC_VAL := $(shell lscpu -p | egrep -v '^\#' | wc -l)
endif

SPARSE_BENCH_DIR = deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = deps/SparseRooflineBenchmark/build

$(SPARSE_BENCH_CLONE): 
	git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH): $(SPARSE_BENCH_CLONE) $(SPARSE_BENCH_DIR)/src/*
	touch $(SPARSE_BENCH)

TACO_DIR = deps/taco
TACO_CLONE = $(TACO_DIR)/.git
TACO = deps/taco/build/lib/libtaco.*
TACO_CXXFLAGS = $(CXXFLAGS) -I$(TACO_DIR)/include -I$(TACO_DIR)/src
TACO_LDLIBS = $(LDLIBS) -L$(TACO_DIR)/build/lib -ltaco -ldl

$(TACO_CLONE): 
	git submodule update --init $(TACO_DIR)

$(TACO): $(TACO_CLONE) $(TACO_DIR)/src/* $(TACO_DIR)/include/*
	cd $(TACO_DIR) ;\
	mkdir -p build ;\
	cd build ;\
	cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release .. ;\
	make taco -j$(NPROC_VAL)

SPMV = spmv/spmv spmv/spmv_taco

all: $(SPMV)

clean:
	rm -rf spmv
	rm -rf *.o *.dSYM *.trace


spmm/spmm_taco: $(SPARSE_BENCH) $(TACO) spmm/spmm_taco.cpp
	$(CXX) $(TACO_CXXFLAGS) -o $@ spmm/spmm_taco.cpp $(TACO_LDLIBS)

spmv/spmv: $(SPARSE_BENCH) spmv/spmv.cpp
	$(CXX) $(CXXFLAGS) -o $@ spmv/spmv.cpp $(LDLIBS)

spmv/spmv_taco: $(SPARSE_BENCH) $(TACO) spmv/spmv_taco.cpp
	$(CXX) $(TACO_CXXFLAGS) -o $@ spmv/spmv_taco.cpp $(TACO_LDLIBS)
