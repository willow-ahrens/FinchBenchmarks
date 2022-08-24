using Finch
using SparseArrays
using BenchmarkTools

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket


function triangle_taco(A)
    b_ref = Scalar{Int32(0)}()
    
    #@finch @loop i j b_ref[i] += A[i, j] * A[i, k] * A[k, j]

    ttwrite("b.ttx", (), [0], ())
    (I, J, V) = findnz(A)
    ttwrite("A1.ttx", (I, J), ones(Int32, length(V)), size(A))
    ttwrite("A2.ttx", (I, J), ones(Int32, length(V)), size(A))
    ttwrite("A3.ttx", (I, J), ones(Int32, length(V)), size(A))

    io = IOBuffer()

    @info :run
    run(pipeline(`./triangle_taco b.ttx A1.ttx A2.ttx A3.ttx`, stdout=io))

    b = fsparse(ttread("b.ttx")...)
    println(b)

    #@assert b() == b_ref()
    return parse(Int64, String(take!(io))) * 1.0e-9
end

#function spmv_finch(_A, x)
#    A = fiber(_A)
#    y = fiber(x)
#    x = fiber(x)
#    println(@finch_code @loop i j y[i] += A[i, j] * x[j])
#    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
#end

@info "loading"
A = SparseMatrixCSC(matrixdepot("Boeing/ct20stif"))
(m, n) = size(A)
#@info "taco"
println("taco_time: ", triangle_taco(A))
#@info "finch"
#println("finch_time: ", spmv_finch(A, rand(n)))
