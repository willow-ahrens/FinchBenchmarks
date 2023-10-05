SPARSE_BENCH_DIR = deps/SparseRooflineBenchmark
SPARSE_BENCH_CLONE = $(SPARSE_BENCH_DIR)/.git
SPARSE_BENCH_BUILD = deps/SparseRooflineBenchmark/build

$(SPARSE_BENCH_CLONE): 
    git submodule update --init $(SPARSE_BENCH_DIR)

$(SPARSE_BENCH_BUILD): $(SPARSE_BENCH_DIR)/src/*
	touch $(SPARSE_BENCH_BUILD)


