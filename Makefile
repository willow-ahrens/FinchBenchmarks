CC = gcc
LD = ld
TACO = taco
CXXFLAGS += -std=c++11 -I$(TACO)/include -I$(TACO)/src
LDLIBS += -L$(TACO)/build/lib -ltaco -ldl

#ARCH = $(shell uname)
#ifeq ($(wildcard $(TOP)/src/Makefile.$(ARCH)),)
#	MYARCH = Default
#else
#	MYARCH = $(ARCH)
#endif
#include Makefile.$(MYARCH)

all: spmv_taco

clean:
	rm -rf spmv_taco
	rm -rf *.o *.dSYM *.trace
	cd taco/build && make clean
	rm -rf opencv/build
	rm -rf opencv/install

spmv_taco: spmv_taco.o taco/build/bin/taco
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

.PHONY: opencv
opencv:
	./build_opencv.sh

taco/build/bin/taco:
	./build_taco.sh