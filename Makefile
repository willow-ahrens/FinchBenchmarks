CC = gcc
CXX = g++
LD = ld
CXXFLAGS += -std=c++17 -O3 -march=native
LDLIBS +=

ifeq ("$(shell uname)","Darwin")
export NPROC_VAL := $(shell sysctl -n hw.logicalcpu_max )
else
export NPROC_VAL := $(shell lscpu -p | egrep -v '^\#' | wc -l)
endif

SPGEMM_EIGEN = spgemm/spgemm_eigen
SPGEMM_MKL = spgemm/spgemm_mkl

SPARSE_BENCH_DIR = deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH = $(SPARSE_BENCH_DIR)/build/hello

TACO_DIR = deps/taco
TACO_CLONE = $(TACO_DIR)/.git
TACO = $(TACO_DIR)/build/hello
TACO_CXXFLAGS = -fopenmp -I$(TACO_DIR)/include -I$(TACO_DIR)/src
TACO_LDLIBS = -L$(TACO_DIR)/build/lib -ltaco -ldl

NACHO_DIR = deps/nacho
NACHO_CLONE = $(NACHO_DIR)/.git
NACHO = $(NACHO_DIR)/build/hello
NACHO_ENV = $(CURDIR)/.nacho-env
NACHO_PYTHON = $(NACHO_ENV)/bin/python
MICROMAMBA_ROOT = $(CURDIR)/.micromamba
MICROMAMBA = $(MICROMAMBA_ROOT)/micromamba

EIGEN_DIR = deps/eigen
EIGEN_CLONE = $(EIGEN_DIR)/.git
EIGEN_CXXFLAGS = -I$(EIGEN_DIR) -fopenmp

NACHO_CXXFLAGS = -std=c++20 -I$(NACHO_DIR)/generated -I$(NACHO_DIR)/runtime -I$(NACHO_ENV)/include
NACHO_LDLIBS = -L$(NACHO_ENV)/lib -ltbb -Wl,-rpath,$(NACHO_ENV)/lib

MKLROOT = /opt/intel/oneapi/mkl/2025.3
MKL_CXXFLAGS = -I$(MKLROOT)/include
MKL_LDLIBS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -liomp5

ALL_TARGETS = $(SPGEMM_EIGEN) spadd/spadd_eigen spadd/spadd_nacho hadamard/hadamard_eigen mttkrp/mttkrp_taco spmspv/spmspv_eigen $(NACHO)

ifeq ($(shell uname -m), x86_64)
	ALL_TARGETS += $(SPGEMM_MKL) spadd/spadd_mkl outer/outer_mkl outer_transpose/outer_mkl
endif

all: $(ALL_TARGETS)

clean:
	rm -f $(ALL_TARGETS)
	rm -rf *.o *.dSYM *.trace

$(SPARSE_BENCH_CLONE):
	git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH): $(SPARSE_BENCH_CLONE)
	mkdir -p $(SPARSE_BENCH) ;\
	touch $(SPARSE_BENCH)

$(TACO_CLONE):
	git submodule update --init $(TACO_DIR)

$(TACO): $(TACO_CLONE)
	cd $(TACO_DIR) ;\
	mkdir -p build ;\
	cd build ;\
	cmake -DOPENMP=ON -DPYTHON=false -DCMAKE_BUILD_TYPE=Release .. ;\
	make taco -j$(NPROC_VAL) ;\
	touch hello

$(NACHO_CLONE):
	git submodule update --init $(NACHO_DIR)

$(MICROMAMBA):
	mkdir -p $(MICROMAMBA_ROOT) && \
	curl -fsSL -o $@ https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64 && \
	chmod +x $@

$(NACHO_PYTHON): $(MICROMAMBA)
	MAMBA_ROOT_PREFIX=$(MICROMAMBA_ROOT) $(MICROMAMBA) create -y -p $(NACHO_ENV) \
		-c nvidia -c conda-forge python=3.12 tbb-devel cmake nvidia::cuda-toolkit=12.6 texlive-core ghostscript

$(NACHO): $(NACHO_CLONE) $(NACHO_PYTHON)
	cd $(NACHO_DIR) && \
	export PATH=$(NACHO_ENV)/bin:$$PATH CMAKE_PREFIX_PATH=$(NACHO_ENV) && \
	python -m pip install -r requirements.txt && \
	python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && \
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && \
	cmake --build build -j$(NPROC_VAL) && \
	./build/compiler && \
	python -m pip install --no-build-isolation -ve . && \
	touch build/hello

$(EIGEN_CLONE):
	git submodule update --init $(EIGEN_DIR)

spgemm/spgemm_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spgemm/spgemm_eigen.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spgemm/spgemm_eigen.cpp

spgemm/spgemm_mkl: $(SPARSE_BENCH) spgemm/spgemm_mkl.cpp
	bash -c 'source /opt/intel/oneapi/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spgemm/spgemm_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'

outer/outer_mkl: $(SPARSE_BENCH) outer/outer_mkl.cpp
	bash -c 'source /opt/intel/oneapi/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ outer/outer_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'

outer_transpose/outer_mkl: $(SPARSE_BENCH) outer_transpose/outer_mkl.cpp
	bash -c 'source /opt/intel/oneapi/setvars.sh; $(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ outer_transpose/outer_mkl.cpp $(LDLIBS) $(MKL_LDLIBS)'

spadd/spadd_mkl: spadd/mkl_impl.cpp
	bash -c 'source /opt/intel/oneapi/setvars.sh; $(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) $(EIGEN_CXXFLAGS) $(MKL_CXXFLAGS) -o $@ spadd/mkl_impl.cpp $(LDLIBS) $(MKL_LDLIBS) $(TACO_LDLIBS)'

spadd/spadd_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spadd/eigen_impl.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spadd/eigen_impl.cpp

$(NACHO_DCSR_ADD): $(NACHO)

spadd/spadd_nacho: $(SPARSE_BENCH) $(EIGEN_CLONE) $(NACHO) $(NACHO_DCSR_ADD) spadd/nacho_impl.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) $(NACHO_CXXFLAGS) -o $@ spadd/nacho_impl.cpp $(NACHO_DIR)/generated/dcsr_add_cpu.cpp $(LDLIBS) $(NACHO_LDLIBS)

hadamard/hadamard_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) hadamard/eigen_impl.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ hadamard/eigen_impl.cpp

mttkrp/mttkrp_taco: $(SPARSE_BENCH) $(TACO) mttkrp/taco_impl.cpp
	$(CXX) $(CXXFLAGS) $(TACO_CXXFLAGS) -o $@ mttkrp/taco_impl.cpp $(LDLIBS) $(TACO_LDLIBS)

spmspv/spmspv_eigen: $(SPARSE_BENCH) $(EIGEN_CLONE) spmspv/spmspv_eigen.cpp
	$(CXX) $(CXXFLAGS) $(EIGEN_CXXFLAGS) -o $@ spmspv/spmspv_eigen.cpp

