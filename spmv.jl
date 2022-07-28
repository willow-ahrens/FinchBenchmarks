using Finch
using SparseArrays

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

    run(pipeline(`./spmv_taco y.ttx A.ttx x.ttx`, stdout=io))

    y = fsparse(ttread("y.ttx")...)

    @assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

spmv_taco(ones(10, 10), ones(10))