CC = gcc
LD = ld
TACO = taco
CXXFLAGS += -std=c++11 -I$(TACO)/include -I$(TACO)/src -DDECIMAL_DIG=17
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

spmv_taco: spmv_taco.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDLIBS)