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

    @info :y
    ttwrite("y.ttx", ffindnz(y_ref)..., size(y_ref))
    (I, J, V) = findnz(A)
    @info :A
    ttwrite("A.ttx", (I, J), V, size(A))
    @info :x
    ttwrite("x.ttx", ffindnz(x)..., size(x))

    io = IOBuffer()

    @info :run
    run(pipeline(`./spmv_taco y.ttx A.ttx x.ttx`, stdout=io))

    y = fsparse(ttread("y.ttx")...)

    @finch @loop i j y_ref[i] += A[i, j] * x[j]

    #@assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmv_finch(_A, x)
    A = fiber(_A)
    y = fiber(x)
    x = fiber(x)
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
end

@info "loading"
A = SparseMatrixCSC(matrixdepot("Boeing/ct20stif"))
(m, n) = size(A)
@info "taco"
println("taco_time: ", spmv_taco(A, rand(n)))
@info "finch"
println("finch_time: ", spmv_finch(A, rand(n)))