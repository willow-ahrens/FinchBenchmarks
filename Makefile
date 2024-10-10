CC = gcc
CXX = g++
LD = ld
CXXFLAGS += -std=c++17
LDLIBS +=

ifeq ("$(shell uname)","Darwin")
export NPROC_VAL := $(shell sysctl -n hw.logicalcpu_max )
else
export NPROC_VAL := $(shell lscpu -p | egrep -v '^\#' | wc -l)
endif

SPMV = spmv/spmv_taco spmv/spmv_taco_row_maj
SPMV_EIGEN = spmv/spmv_eigen
SPMV_MKL = spmv/spmv_mkl

SPGEMM = spgemm/spgemm_taco
SPGEMM_EIGEN = spgemm/spgemm_eigen
SPGEMM_MKL = spgemm/spgemm_mkl

all: $(SPMV) $(SPGEMM) $(SPMV_EIGEN) $(SPGEMM_EIGEN) $(SPMV_MKL) $(SPGEMM_MKL)

SPARSE_BENCH_DIR = deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = deps/SparseRooflineBenchmark/build/hello

$(SPARSE_BENCH_CLONE): 
	git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH): $(SPARSE_BENCH_CLONE)
	mkdir -p $(SPARSE_BENCH) ;\
	touch $(SPARSE_BENCH)


TACO_DIR = deps/taco
EIGEN_DIR = deps/eigen-3.4.0
TACO_CLONE = $(TACO_DIR)/.git
TACO = deps/taco/build/lib/libtaco.*
TACO_CXXFLAGS = $(CXXFLAGS) -I$(TACO_DIR)/include -I$(TACO_DIR)/src
EIGEN_CXXFLAGS = $(CXXFLAGS) -I$(EIGEN_DIR)
TACO_LDLIBS = $(LDLIBS) -L$(TACO_DIR)/build/lib -ltaco -ldl

MKLROOT = /data/scratch/changwan/mkl/2024.2
MKL_CXXFLAGS = $(CXXFLAGS) -I$(MKLROOT)/include
MKL_LDLIBS = $(LDLIBS) -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm -liomp5

$(TACO_CLONE): 
	git submodule update --init $(TACO_DIR)

$(TACO): $(TACO_CLONE)
	cd $(TACO_DIR) ;\
	mkdir -p build ;\
	cd build ;\
	cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release .. ;\
	make taco -j$(NPROC_VAL)

clean:
	rm -f $(SPMV) $(SPGEMM) $(SPMV_EIGEN) $(SPGEMM_EIGEN) $(SPMV_MKL) $(SPGEMM_MKL)
	rm -rf *.o *.dSYM *.trace

spgemm/spgemm_taco: $(SPARSE_BENCH) $(TACO) spgemm/spgemm_taco.cpp
	$(CXX) $(TACO_CXXFLAGS) -o $@ spgemm/spgemm_taco.cpp $(TACO_LDLIBS)

spgemm/spgemm_eigen: $(SPARSE_BENCH) spgemm/spgemm_eigen.cpp
	$(CXX) $(EIGEN_CXXFLAGS) -o $@ spgemm/spgemm_eigen.cpp 

spmv/spmv_taco: $(SPARSE_BENCH) $(TACO) spmv/spmv_taco.cpp
	$(CXX) $(TACO_CXXFLAGS) -o $@ spmv/spmv_taco.cpp $(TACO_LDLIBS)

spmv/spmv_taco_row_maj: $(SPARSE_BENCH) $(TACO) spmv/spmv_taco_row_maj.cpp
	$(CXX) $(TACO_CXXFLAGS) -o $@ spmv/spmv_taco_row_maj.cpp $(TACO_LDLIBS)

spmv/spmv_eigen: $(SPARSE_BENCH) spmv/spmv_eigen.cpp
	$(CXX) $(EIGEN_CXXFLAGS) -o $@ spmv/spmv_eigen.cpp

spmv/spmv_mkl: $(SPARSE_BENCH) spmv/spmv_mkl.cpp
	$(CXX) $(MKL_CXXFLAGS) -o $@ spmv/spmv_mkl.cpp $(MKL_LDLIBS)

spgemm/spgemm_mkl: $(SPARSE_BENCH) spgemm/spgemm_mkl.cpp
	$(CXX) $(MKL_CXXFLAGS) -o $@ spgemm/spgemm_mkl.cpp $(MKL_LDLIBS)
