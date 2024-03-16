using Finch
using TensorMarket
using JSON
function spgemm_taco(args, A, B)
    mktempdir(@__DIR__, prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A)) #TACO matrix market readerr can only read real-valued matrices
        fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B)) #TACO matrix market readerr can only read real-valued matrices
        taco_path = joinpath(@__DIR__, "../deps/taco/build/lib")
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"$taco_path", "LD_LIBRARY_PATH" => "$taco_path", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
            spgemm_path = joinpath(@__DIR__, "spgemm_taco")
            run(`$spgemm_path -i $tmpdir -o $tmpdir -- $args`)
        end
        C = fread(C_path)
        time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time*10^-9, C=C)
    end
end

spgemm_taco_inner(A, B) = spgemm_taco(`--schedule inner`, A, B)
spgemm_taco_outer(A, B) = spgemm_taco(`--schedule outer`, A, B)
spgemm_taco_gustavson(A, B) = spgemm_taco(`--schedule gustavson`, A, B)
