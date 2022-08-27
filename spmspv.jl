using Finch
using SparseArrays
using BenchmarkTools
using Scratch

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket


function spmspv_taco(_A, x, key)
    y_ref = @fiber(d(e(0.0)))
    A = fiber(_A)
    @finch @loop i j y_ref[i] += A[i, j] * x[j]
    @finch @loop i y_ref[i] = 0

    y_file = joinpath(mktempdir(prefix="spmspv_taco_$(key)"), "y.ttx")
    persist_dir = joinpath(get_scratch!("Finch-CGO-2023"), "spmspv_taco_$(key)")
    mkpath(persist_dir)
    A_file = joinpath(persist_dir, "A.ttx")
    x_file = joinpath(mktempdir(prefix="spmspv_taco_$(key)"), "x.ttx")

    ttwrite(y_file, ffindnz(y_ref)..., size(y_ref))
    ttwrite(x_file, ffindnz(x)..., size(x))
    #if !(isfile(A_file))
        ((I, J), V) = ffindnz(A)
        ttwrite(A_file, (I, J), V, size(_A))
    #end

    io = IOBuffer()

    println("running")

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./taco/build/lib", "LD_LIBRARY_PATH" => "./taco/build/lib") do
        run(pipeline(`./spmspv_taco $y_file $A_file $x_file`, stdout=io))
    end

    y = fsparse(ttread(y_file)...)

    @finch @loop i j y_ref[i] += A[i, j] * x[j]

    #println((FiberArray(y)[1:10], FiberArray(y_ref)[1:10]))
    #@assert FiberArray(y) ≈ FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmspv_finch(_A, x)
    A = fiber(_A)
    x = copyto!(@fiber(sl(e(0.0))), x)
    y = @fiber(d(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
end

function spmspv_gallop_finch(_A, x)
    A = fiber(_A)
    x = copyto!(@fiber(sl(e(0.0))), x)
    y = @fiber(d(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j::gallop] * x[j::gallop])
end

function spmspv_lead_finch(_A, x)
    A = fiber(_A)
    x = copyto!(@fiber(sl(e(0.0))), x)
    y = @fiber(d(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j::gallop] * x[j])
end

function spmspv_follow_finch(_A, x)
    A = fiber(_A)
    x = copyto!(@fiber(sl(e(0.0))), x)
    y = @fiber(d(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j::gallop])
end

function spmspv_finch_vbl(_A, x)
    A = copyto!(@fiber(d(sv(e(0.0)))), fiber(_A))
    x = copyto!(@fiber(sl(e(0.0))), x)
    y = @fiber(d(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j::gallop] * x[j])
end

function main()
    for (mtx, key) in [
        ("Boeing/ct20stif", "ct20stif"),
        ("HB/cegb2802", "cegb2802"),
    ]
        A = SparseMatrixCSC{Float64}(matrixdepot(mtx))
        (m, n) = size(A)
        println((key, m, n, nnz(A)))
        taco_uniform_times = []
        finch_uniform_times = []
        finch_lead_uniform_times = []
        finch_follow_uniform_times = []
        finch_vbl_uniform_times = []
        for x = 1:100
            x = fsprand((n,), 0.1)
            taco_time = spmspv_taco(A, x, key)
            println("taco_time ", taco_time)
            push!(taco_uniform_times, taco_time)
            finch_time = spmspv_finch(A, x)
            println("finch_time ", finch_time)
            push!(finch_uniform_times, finch_time)
            finch_gallop_time = spmspv_gallop_finch(A, x)
            println("finch_gallop_time ", finch_time)
            push!(finch_gallop_uniform_times, finch_gallop_time)
            finch_lead_time = spmspv_lead_finch(A, x)
            println("finch_lead_time ", finch_time)
            push!(finch_lead_uniform_times, finch_lead_time)
            finch_follow_time = spmspv_follow_finch(A, x)
            println("finch_follow_time ", finch_time)
            push!(finch_follow_uniform_times, finch_follow_time)
            finch_vbl_time = spmspv_finch_vbl(A, x)
            println("finch_vbl_time ", finch_vbl_time)
            push!(finch_vbl_uniform_times, finch_vbl_time)
        end
    end
end

main()