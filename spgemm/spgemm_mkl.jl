using Finch
using TensorMarket
using JSON
function spgemm_mkl(A, B)
    tmpdir = mktempdir(@__DIR__, prefix="tmp_")
    A_path = joinpath(tmpdir, "A.ttx")
    B_path = joinpath(tmpdir, "B.ttx")
    C_path = joinpath(tmpdir, "C.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A)) #TACO matrix market readerr can only read real-valued matrices
    fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B)) #TACO matrix market readerr can only read real-valued matrices
    mkl_path = joinpath(@__DIR__, "/data/scratch/changwan/mkl/2024.2/lib/intel64")
    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$mkl_path", "LD_LIBRARY_PATH" => "$mkl_path", "MKL_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
	    spgemm_path = joinpath(@__DIR__, "spgemm_mkl")
	    run(`$spgemm_path -i $tmpdir -o $tmpdir`)
    end
    C = fread(C_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, C=C)
end

