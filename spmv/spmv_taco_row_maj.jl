using Finch
using TensorMarket
using JSON
function spmv_taco_helper_row_maj(args, A, x)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        x_path = joinpath(tmpdir, "x.ttx")
        y_path = joinpath(tmpdir, "y.ttx")
        A_T = Tensor(Dense(SparseList(Element(0.0))))
        @finch mode=:fast begin
            A_T .= 0
            for j=_, i=_
                A_T[i, j] = A[j, i]
            end
        end
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A_T))
        fwrite(x_path, Tensor(Dense(Element(0.0)), x))
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
            spmv_path = joinpath(@__DIR__, "spmv_taco_row_maj")
            run(`$spmv_path -i $tmpdir -o $tmpdir $args`)
        end
        y = fread(y_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time*10^-9, y=y)
    end
end

spmv_taco_row_maj(y, A, x) = spmv_taco_helper_row_maj("", A, x)
