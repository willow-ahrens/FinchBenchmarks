using Finch
using TensorMarket
using JSON
function spgemm_taco(args, A, B)
    mktempdir(prefix="input_") do tmpdir
        A_path = joinpath(tmpdir, "A.ttx")
        B_path = joinpath(tmpdir, "B.ttx")
        C_path = joinpath(tmpdir, "C.ttx")
        fwrite(A_path, fiber(A))
        fwrite(B_path, fiber(B))
        run(`./spgemm_taco -i=$tmpdir -o=$tmpdir $args`)
        C = fread(C_path)
        time = JSON.read(joinpath(tmpdir, "measurements.json"))["time"]
        return (;time=time, C=C)
    end
end

spgemm_taco_inner(A, B) = spgemm_taco("", A, B)