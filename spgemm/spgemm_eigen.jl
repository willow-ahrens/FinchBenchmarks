using Finch
using TensorMarket
using JSON
function spgemm_eigen(A, B)
    tmpdir = mktempdir(@__DIR__, prefix="tmp_")
    A_path = joinpath(tmpdir, "A.ttx")
    B_path = joinpath(tmpdir, "B.ttx")
    C_path = joinpath(tmpdir, "C.ttx")
    fwrite(A_path, Tensor(Dense(SparseList(Element(0.0))), A)) #TACO matrix market readerr can only read real-valued matrices
    fwrite(B_path, Tensor(Dense(SparseList(Element(0.0))), B)) #TACO matrix market readerr can only read real-valued matrices
    eigen_path = joinpath(@__DIR__, "../deps/eigen-3.4.0")
    withenv("PATH" => "$eigen_path", "EIGEN_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        spgemm_path = joinpath(@__DIR__, "spgemm_eigen")
        run(`$spgemm_path -i $tmpdir -o $tmpdir`)
    end
    C = fread(C_path)
    time = JSON.parsefile(joinpath(tmpdir, "measurements.json"))["time"]
    return (;time=time*10^-9, C=C)
end

