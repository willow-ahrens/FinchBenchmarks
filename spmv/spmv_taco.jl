using Finch
using TensorMarket
using JSON
function spmv_taco_helper(args, A, x)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        x_path = joinpath(tmpdir, "x.ttx")
        y_path = joinpath(tmpdir, "y.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A))
        fwrite(x_path, Tensor(Dense(Element(0.0)), x))
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"../deps/taco/build/lib", "LD_LIBRARY_PATH" => "../deps/taco/build/lib", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
            run(`./spmv_taco -i $tmpdir -o $tmpdir $args`)
        end
        y = fread(y_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time*10^-9, y=y)
    end
end

spmv_taco(y, A, x) = spmv_taco_helper("", A, x)
