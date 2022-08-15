CC = gcc
LD = ld
TACO = taco
TACORLE = taco-rle
OPENCV = opencv
CXXFLAGS += -std=c++11 -I$(TACO)/include -I$(TACO)/src
LDLIBS += -L$(TACO)/build/lib -ltaco -ldl

CXXFLAGS_TACORLE += -std=c++11 -I$(TACORLE)/include -I$(TACORLE)/src
LDLIBS_TACORLE += -L$(TACORLE)/build/lib -ltaco -ldl

CXXFLAGS_CV += -std=c++11 -I$(OPENCV)/install/include/opencv4 -I$(OPENCV)/install/include/opencv4/opencv2
LDLIBS_CV += -L$(OPENCV)/install/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_imgcodecs 

TACOBUILD = $(TACO)/build/lib/libtaco.*
TACORLEBUILD = $(TACORLE)/build/lib/libtaco.*
OPENCVBUILD = $(OPENCV)/build/lib/libopencv_core.*

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

all: spmv_taco alpha_taco_rle alpha_opencv

clean:
	rm -rf spmv_taco
	rm -rf alpha_opencv
	rm -rf *.o *.dSYM *.trace
	# rm -rf $(TACO)/build
	# rm -rf $(TACORLE)/build
	# rm -rf $(OPENCV)/build
	# rm -rf $(OPENCV)/install

spmv_taco: spmv_taco.o $(TACOBUILD)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)

alpha_taco_rle: alpha_taco_rle.cpp $(TACORLEBUILD)
	$(CXX) $(CXXFLAGS_TACORLE) -o $@ $< $(LDLIBS_TACORLE)

alpha_opencv: alpha_opencv.cpp $(OPENCVBUILD)
	$(CXX) $(CXXFLAGS_CV) -o $@ $< $(LDLIBS_CV)

all_pairs_opencv: all_pairs_opencv.cpp $(OPENCVBUILD)
	$(CXX) $(CXXFLAGS_CV) -o $@ $< $(LDLIBS_CV)

$(OPENCVBUILD):
	mkdir -p opencv/build
	mkdir -p opencv/install
	cd opencv/build
	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=../install -DBUILD_ZLIB=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_PNG=ON .. 
	make -j$(NPROC_VAL)
	make install

$(TACOBUILD):
	cd $(TACO)
	mkdir -p build
	cd build
	cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release ..
	make -j$(NPROC_VAL)

$(TACORLEBUILD):
	cd $(TACORLE)
	mkdir -p build
	cd build
	cmake -DPYTHON=false -DCMAKE_BUILD_TYPE=Release ..
	make -j$(NPROC_VAL)