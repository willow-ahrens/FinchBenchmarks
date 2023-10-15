using Finch
using TensorMarket
using JSON
function spgemm_taco(args, A, B)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        fwrite(A_path, Fiber!(Dense(SparseList(Element(0.0))), A)) #TACO matrix market readerr can only read real-valued matrices
        fwrite(B_path, Fiber!(Dense(SparseList(Element(0.0))), B)) #TACO matrix market readerr can only read real-valued matrices
        withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"../deps/taco/build/lib", "LD_LIBRARY_PATH" => "../deps/taco/build/lib", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        run(`./spgemm_taco -i $tmpdir -o $tmpdir $args`)
        end
        C = fread(C_path)
        time = JSON.read(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time, C=C)
    end
end

spgemm_taco_inner(A, B) = spgemm_taco("", A, B)