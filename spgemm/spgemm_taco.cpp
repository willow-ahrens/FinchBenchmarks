#include "taco.h"
//#include "taco/format.h"
//#include "taco/lower/lower.h"
//#include "taco/ir/ir.h"
#include <chrono>
#include <sys/stat.h>
#include <iostream>
#include <cstdint>
#include "../deps/SparseRooflineBenchmark/src/benchmark.hpp"

//namespace fs = std::__fs::filesystem;

using namespace taco;


//format = 0: DD, 1:DS, 2:SS
//computemode = 0:Gustavson, 1:inner proudct, 2:outer product

#define FORMAT_DD (0)
#define FORMAT_DS (1)
#define FORMAT_SS (2)
#define MOD_GUS (0)
#define MOD_INNER (1)
#define MOD_OUTER (2)

void experiment(std::string input, std::string output, int Aformat, int Bformat, int Cformat, int computemode){

	//fprintf(stderr, "OOO: %d %d %d %d\n", Aformat, Bformat, Cformat, computemode);
	// assume sym (will be lifted)   
    Tensor<double> A;
    Tensor<double> B;
    switch(Aformat) {
	    case FORMAT_DD:
		    A = read(input+".ttx", Format({Dense, Dense}), true);
		    break;
	    case FORMAT_DS:
		    A = read(input+".ttx", Format({Dense, Sparse}), true);
		    break;
	    case FORMAT_SS:
		    A = read(input+".ttx", Format({Sparse, Sparse}), true);
		    break;
    }
    switch(Bformat) {
	    case FORMAT_DD:
		    B = read(input+"_s.ttx", Format({Dense, Dense}), true);
		    break;
	    case FORMAT_DS:
		    B = read(input+"_s.ttx", Format({Dense, Sparse}), true);
		    break;
	    case FORMAT_SS:
		    B = read(input+"_s.ttx", Format({Sparse, Sparse}), true);
		    break;
    }

    int m = A.getDimension(0);
    int mn = A.getDimension(1);
    int n = B.getDimension(1);

    auto cF = Format({Dense,Dense});
    switch(Cformat) {
	    case FORMAT_DD:
		    break;
	    case FORMAT_DS:
		    cF = Format({Dense, Sparse});
		    break;
	    case FORMAT_SS:
		    cF = Format({Sparse, Sparse});
		    break;
    }
    Tensor<double> C("C", {m, n}, cF); // cond

    //fprintf(stderr, "OOOO\n");
	//std::cerr<<"OO: "<<cF<<std::endl;

    IndexVar i, j, k;
    IndexStmt stmt;

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
    //Gustavson
    C(i, j) += A(i, k) * B(k, j);
    //inner product
    //C(i, j) += A(i, k) * B(j, k);
    //outer product
    //C(i, j) += A(k, i) * B(k, j);

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

    //write("C.ttx", C);
//C.printAssembleIR(std::cout, true, true);
C.printComputeIR(std::cout, true, true);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(output+".json");
    measurements_file << measurements;
    measurements_file.close();
}

/*
void experiment(std::string input, std::string output, int verbose){
    Tensor<double> A = read(input+"/A.ttx", Format({Dense, Sparse}), true);
    Tensor<double> x = read(input+"/x.ttx", Format({Dense}), true);
    int m = A.getDimension(0);
    int n = A.getDimension(1);
    Tensor<double> y("y", {n}, Format({Dense}));

    IndexVar i, j;

    y(i) += A(i, j) * x(j);

    //perform an spmv of the matrix in c++

    y.compile();

    // Assemble output indices and numerically compute the result
    auto time = benchmark(
      [&y]() {
        y.setNeedsAssemble(true);
        y.setNeedsCompute(true);
      },
      [&y]() {
        y.assemble();
        y.compute();
      }
    );

    write("y.ttx", y);

    json measurements;
    measurements["time"] = time;
    measurements["memory"] = 0;
    std::ofstream measurements_file(output+"/measurements.json");
    measurements_file << measurements;
    measurements_file.close();
}*/

