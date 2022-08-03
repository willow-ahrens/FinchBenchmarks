CC = gcc
LD = ld
TACO = taco
OPENCV = opencv
CXXFLAGS += -std=c++11 -I$(TACO)/include -I$(TACO)/src
LDLIBS += -L$(TACO)/build/lib -ltaco -ldl

CXXFLAGS_CV += -std=c++11 -I$(OPENCV)/install/include/opencv4 -I$(OPENCV)/install/include/opencv4/opencv2
LDLIBS_CV += -L$(OPENCV)/install/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_imgcodecs 

#ARCH = $(shell uname)
#ifeq ($(wildcard $(TOP)/src/Makefile.$(ARCH)),)
#	MYARCH = Default
#else
#	MYARCH = $(ARCH)
#endif
#include Makefile.$(MYARCH)

ifeq ("$(shell uname)","Darwin")
export NPROC_VAL := $(shell sysctl -n hw.logicalcpu_max )
else
export NPROC_VAL := $(shell lscpu -p | egrep -v '^\#' | wc -l)
endif

all: spmv_taco

clean:
	rm -rf spmv_taco
	rm -rf alpha_opencv
	rm -rf *.o *.dSYM *.trace
	# rm -rf taco/build
	# rm -rf opencv/build
	# rm -rf opencv/install

spmv_taco: spmv_taco.o taco/build/bin/taco
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

alpha_opencv: alpha_opencv.cpp opencv
	$(CXX) $(CXXFLAGS_CV) -o $@ $< $(LDLIBS_CV)

.PHONY: opencv
opencv:
	./build_opencv.sh

taco/build/bin/taco:
	./build_taco.sh