using Finch
using SparseArrays
using BenchmarkTools

include("TensorMarket.jl")
using .TensorMarket


function spmv_taco(A, x)
    y_ref = fiber(x)
    @index @loop i j y_ref[i] += A[i, j] * x[j]
    @index @loop i y_ref[i] = 0

    ttwrite("A.ttx", ffindnz(A)..., size(A))
    ttwrite("x.ttx", ffindnz(x)..., size(x))
    ttwrite("y.ttx", ffindnz(y_ref)..., size(y_ref))

    io = IOBuffer()

    run(pipeline(`./spmv_taco x.ttx A.ttx y.ttx`, stdout=io))

    y = fsparse(ttread("y.ttx")...)

    @index @loop i j y_ref[i] += A[i, j] * x[j]

    @assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmv_finch(A, x)
    A = fiber(A)
    y = fiber(x)
    x = fiber(x)
    return @btime (A = $A; x = $x, y = $y; @index @loop i j y[i] += A[i, j] * x[j])
end

println("taco_time: ", spmv_taco(ones(10, 10), ones(10)))
println("finch_time: ", spmv_finch(ones(10, 10), ones(10)))