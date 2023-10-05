CC = gcc
LD = ld
CXXFLAGS += -std=c++20
LDLIBS +=

all: spmv

clean:
	rm -rf spmv
	rm -rf *.o *.dSYM *.trace

spmv: spmv.o
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDLIBS)