using Finch
using SparseArrays
using BenchmarkTools
using Scratch

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket


function spmspv_taco(_A, x, key)
    y_ref = fiber(x)
    A = fiber(_A)
    @finch @loop i j y_ref[i] += A[i, j] * x[j]
    @finch @loop i y_ref[i] = 0

    y_file = joinpath(mktempdir(prefix="spmv_taco_$(key)"), "y.ttx")
    persist_dir = joinpath(get_scratch!("Finch-CGO-2023"), "spmv_taco_$(key)")
    mkpath(persist_dir)
    A_file = joinpath(persist_dir, "A.ttx")
    x_file = joinpath(persist_dir, "x.ttx")

    ttwrite(y_file, ffindnz(y_ref)..., size(y_ref))
    #if !(isfile(A_file) && isfile(x_file))
        (I, J, V) = findnz(_A)
        ttwrite(A_file, (I, J), V, size(_A))
        ttwrite(x_file, ffindnz(x)..., size(x))
    #end

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./taco/build/lib", "LD_LIBRARY_PATH" => "./taco/build/lib") do
        run(pipeline(`./spmv_taco $y_file $A_file $x_file`, stdout=io))
    end

    #y = fsparse(ttread(y_file)...)

    #@finch @loop i j y_ref[i] += A[i, j] * x[j]

    #@assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmv_finch(_A, x)
    A = fiber(_A)
    y = fiber(x)
    x = fiber(x)
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
end

function spmv_finch_vbl(_A, x)
    A = copyto!(@fiber(d(sv(e(0.0)))), fiber(_A))
    y = fiber(x)
    x = fiber(x)
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
end

function main()
    for (mtx, key) in [
        ("Boeing/ct20stif", "ct20stif"),
    ]
    for p in [0.1, 0.0001]
        A = SparseMatrixCSC(matrixdepot(mtx))
        (m, n) = size(A)
        println((key, m, n, nnz(A)))
        x = fsprand(n, p)
        taco_time = spmv_taco(A, x, key)
        finch_time = spmv_finch(A, x)
        finch_vbl_time = spmv_finch_vbl(A, x)
        
    end
end

main()