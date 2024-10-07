all: vbatch_gemm

MKL_PATH=/opt/intel/oneapi/mkl/2021.3.0/
# MKL_PATH=/opt/intel/mkl/

%: %.cpp
	g++ -std=c++11 -o $@ -I${MKL_PATH}/include/ $< -L${MKL_PATH}/lib/intel64/ -lmkl_rt

clean:
	rm vbatch_gemm
