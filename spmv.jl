using Finch
using SparseArrays
using BenchmarkTools

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket


function spmv_taco(A, x)
    y_ref = fiber(x)
    @finch @loop i j y_ref[i] += A[i, j] * x[j]
    @finch @loop i y_ref[i] = 0

    y_file = joinpath(mktempdir(prefix="spmv_taco_$(key)"), "y.ttx")
    persist_dir = joinpath(get_scratch!("Finch-CGO-2023"), "spmv_taco_$(key)")
    mkpath(persist_dir)
    A_file = joinpath(persist_dir, "A.ttx")
    x_file = joinpath(persist_dir, "x.ttx")

    ttwrite(y_file, (), [0], ())
    if !(isfile(A_file) && isfile(x_file))
        (I, J, V) = findnz(A)
        ttwrite(A_file, (I, J), ones(Int32, length(V)), size(A))
        ttwrite("x.ttx", ffindnz(x)..., size(x))
    end

    io = IOBuffer()

    @info :run
    run(pipeline(`./spmv_taco $y_file $A_file $x_file`, stdout=io))

    #y = fsparse(ttread(y_file)...)

    #@finch @loop i j y_ref[i] += A[i, j] * x[j]

    #@assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmv_finch(_A, x)
    A = fiber(_A)
    y = fiber(x)
    x = fiber(x)
    println(@finch_code @loop i j y[i] += A[i, j] * x[j])
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
end

function spmv_finch_vbl(_A, x)
    A = copyto!(@fiber(d(sv(e(0.0)))), fiber(_A))
    y = fiber(x)
    x = fiber(x)
    println(@finch_code @loop i j y[i] += A[i, j] * x[j])
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
end

@info "loading"

A = SparseMatrixCSC(matrixdepot("Boeing/ct20stif"))
(m, n) = size(A)
@info "taco"
println("taco_time: ", spmv_taco(A, rand(n)))
@info "finch"
println("finch_time: ", spmv_finch(A, rand(n)))
@info "finch_vbl"
println("finch_vbl_time: ", spmv_finch_vbl(A, rand(n)))