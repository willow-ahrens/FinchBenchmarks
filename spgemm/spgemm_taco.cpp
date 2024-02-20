#include "taco.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

using namespace taco;

//format = 0: DD, 1:DS, 2:SS
//computemode = 0:Gustavson, 1:inner proudct, 2:outer product

#define FORMAT_DD (0)
#define FORMAT_DS (1)
#define FORMAT_SS (2)
#define MOD_GUS (0)
#define MOD_INNER (1)
#define MOD_OUTER (2)

int main(int argc, char ** argv){
  auto params = parse(argc, argv);

	//fprintf(stderr, "OOO: %d %d %d %d\n", Aformat, Bformat, Cformat, computemode);
	// assume sym (will be lifted)   
    Tensor<double> A;
    Tensor<double> B;
    A = read(params.input+"/A.ttx", Format({Dense, Sparse}), true);
    B = read(params.input+"/B.ttx", Format({Dense, Sparse}), true);

    int m = A.getDimension(0);
    int n = B.getDimension(1);

    Tensor<double> C("C", {m, n}, Format({Dense, Sparse})); // cond

    //fprintf(stderr, "OOOO\n");
	//std::cerr<<"OO: "<<cF<<std::endl;

    IndexVar i, j, k;
    IndexStmt stmt;

    /*
    switch(computemode) {
	    case MOD_GUS:
		    C(i, j) += A(i, k) * B(k, j);
		    break;
	    case MOD_INNER:
		    C(i, j) += A(i, k) * B(j, k);
		    stmt= C.getAssignment().concretize();
		    stmt = stmt.reorder({i,j,k}); 
		    break;
	    case MOD_OUTER:
		    C(i, j) += A(k, i) * B(k, j);
		    stmt= C.getAssignment().concretize();
		    stmt = stmt.reorder({k,i,j}); 
		    break;	
    }
    */

    //Gustavson
    C(i, j) += A(i, k) * B(k, j);

    //IndexStmt stmt = C.getAssignment().concretize();
    //stmt = stmt.reorder({i,j,k}); //inner
    //stmt = stmt.reorder({k,i,j}); //outer
    //stmt = stmt.parallelize(i,ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);

    //perform an spmv of the matrix in c++

    C.compile();

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
      [&C]() {
        C.setNeedsAssemble(true);
        C.setNeedsCompute(true);
      },
      [&C]() {
        C.assemble(); //no need for dense ouptut
        C.compute();
      }
    );

    write(params.output+"/C.ttx", C);
    //C.printAssembleIR(std::cout, true, true);
    //C.printComputeIR(std::cout, true, true);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(params.output+"/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
    return 0;
}