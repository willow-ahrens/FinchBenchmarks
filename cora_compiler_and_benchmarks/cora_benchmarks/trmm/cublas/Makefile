CC=nvcc

SRCS=$(wildcard *.cu)
OBJS=$(SRCS:.cu=.o)
# CUDAFLAGS=-G -g -lineinfo -arch=compute_${CUDA_ARCH} -code=sm_${CUDA_ARCH} -D_FORCE_INLINES --use_fast_math -lcublas
CUDAFLAGS=-O3 -arch=compute_${CUDA_ARCH} -code=sm_${CUDA_ARCH} -D_FORCE_INLINES --use_fast_math -lcublas --ptxas-options='-v -warn-lmem-usage -warn-spills' --nvlink-options='-v'

gemm_cublas: gemm_cublas.cu utils.h
	$(CC) -o $@ $< $(CUDAFLAGS)

clean:
	rm gemm_cublas
