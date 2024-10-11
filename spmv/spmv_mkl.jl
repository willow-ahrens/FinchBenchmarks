using Finch
using TensorMarket
using JSON
function spmv_mkl(y, A, x)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        x_path = joinpath(tmpdir, "x.ttx")
        y_path = joinpath(tmpdir, "y.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(x_path, Tensor(Dense(Element(0.0)), x))
	mkl_path = joinpath(@__DIR__, "/data/scratch/changwan/mkl/2024.2/lib/intel64")
	withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$mkl_path", "LD_LIBRARY_PATH" => "$mkl_path", "MKL_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
	  spmv_path = joinpath(@__DIR__, "spmv_mkl")
	  run(`$spmv_path -i $tmpdir -o $tmpdir`)
	end 
        y = fread(y_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time*10^-9, y=y)
    end
end
