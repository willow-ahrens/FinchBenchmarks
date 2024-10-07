FLAGS=-O3 --use_fast_math
# FLAGS=-lineinfo

all: taco_csr_trmm taco_csr_tradd taco_csr_trmul taco_bcsr_trmm taco_bcsr_trmul

taco_csr_trmm: taco_csr_trmm.cu utils.cuh
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_csr_trmm taco_csr_trmm.cu -ltaco

taco_csr_tradd: taco_csr_tradd.cu utils.cuh
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_csr_tradd taco_csr_tradd.cu -ltaco

taco_csr_trmul: taco_csr_trmul.cu utils.cuh
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_csr_trmul taco_csr_trmul.cu -ltaco

taco_bcsr_trmul: taco_bcsr_trmul.cu utils.cuh
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_bcsr_trmul taco_bcsr_trmul.cu -ltaco

taco_bcsr_trmm: taco_bcsr_trmm.cu utils.cuh
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_bcsr_trmm taco_bcsr_trmm.cu -ltaco

clean:
	rm -f taco_csr_trmm taco_csr_tradd taco_bcsr_trmul taco_csr_trmul taco_bcsr_trmm
