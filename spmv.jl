using Finch
using SparseArrays
using BenchmarkTools
using Scratch

using Images, FileIO, FixedPointNumbers, Colors

using MatrixDepot
using TensorDepot
using TensorMarket

const MyInt = Int

include("./generated_code/row_pb_spmv.jl")
include("./generated_code/col_pb_spmv.jl")



function MatrixDepot.downloadcommand(url::AbstractString, filename::AbstractString="-")
    `sh -c 'curl -k "'$url'" -Lso "'$filename'"'`
end

# MatrixDepot.init()

global cold_cache=true

global dummySize=6000000
global dummyA=[]
global dummyB=[]

@noinline
function clear_cache(cold)
    if cold
        global dummySize
        global dummyA
        global dummyB

        ret = 0.0
        if length(dummyA) == 0
            dummyA = Array{Float64}(undef, dummySize)
            dummyB = Array{Float64}(undef, dummySize)
        end
        for i in 1:100 
            dummyA[rand(1:dummySize)] = rand(Int64)/typemax(Int64)
            dummyB[rand(1:dummySize)] = rand(Int64)/typemax(Int64)
        end
        for i in 1:dummySize
            ret += dummyA[i] * dummyB[i];
        end
        return ret
    end
    return 0
end

function sizeof1(x, depth=0)
    s = sizeof(x)
    for n in fieldnames(typeof(x))
    #   println("[depth $depth] x: $x, field: $n")
      field = getfield(x,n)
    #   println("got field: $field")
      s += sizeof1(field, depth+1)
    end
    return s
  end

function spmv_taco(_A, x, key)
    y_ref = @fiber(d(e(0.0)))
    A = fiber(_A)
    @finch @loop i j y_ref[i] += A[i, j] * x[j]
    @finch @loop i y_ref[i] = 0

    y_file = joinpath(mktempdir(prefix="spmv_taco_$(key)"), "y.ttx")
    persist_dir = joinpath(get_scratch!("Finch-CGO-2023"), "spmv_taco_$(key)")
    mkpath(persist_dir)
    A_file = joinpath(persist_dir, "A.ttx")
    x_file = joinpath(mktempdir(prefix="spmv_taco_$(key)"), "x.ttx")

    ttwrite(y_file, ffindnz(y_ref)..., size(y_ref))
    ttwrite(x_file, ffindnz(x)..., size(x))
    if !(isfile(A_file))
        ((I, J), V) = ffindnz(A)
        ttwrite(A_file, (I, J), V, size(_A))
    end

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./taco/build/lib", "LD_LIBRARY_PATH" => "./taco/build/lib", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        run(pipeline(`./spmv_taco $y_file $A_file $x_file`, stdout=io))
    end

    y = fsparse(ttread(y_file)...)

    @finch @loop i j y_ref[i] += A[i, j] * x[j]

    @assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmv_finch_dense(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(d{MyInt}(e(zero(val_type))))), fiber(_A))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j y[i] += A[i, j] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_dense_col(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(d{MyInt}(e(zero(val_type))))), fiber(transpose(_A)))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop j i y[i] += A[j, i] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y

end

function spmv_finch(_A, x, val_type)
    global cold_cache
    A = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(zero(val_type))))), fiber(_A))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j y[i] += A[i, j] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_col(_A, x, val_type)
    global cold_cache
    A = dropdefaults!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(zero(val_type))))), fiber(transpose(_A)))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop j i y[i] += A[j, i] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y

end

function spmv_finch_vbl(_A, x, val_type)
    global cold_cache
    A = dropdefaults!(@fiber(d{MyInt}(sv{MyInt, MyInt}(e(zero(val_type))))), fiber(_A))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j y[i] += A[i, j] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_vbl_col(_A, x, val_type)
    global cold_cache
    A = dropdefaults!(@fiber(d{MyInt}(sv{MyInt, MyInt}(e(zero(val_type))))), fiber(transpose(_A)))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop j i y[i] += A[j, i] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_rl(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(rl{zero(val_type), MyInt, MyInt}())), fiber(_A))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j y[i] += A[i, j] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    # time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j (@sieve (Av != 0) (y[i] += (Av[] * x[j]))) where (Av[] = A[i,j])) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_rl_col(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(rl{zero(val_type), MyInt, MyInt}())), fiber(transpose(_A)))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop j i y[i] += A[j, i] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    # time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch  @loop j i (@sieve (Av != 0) (y[i] += (Av[] * x[j]))) where (Av[] = A[j,i])) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_packbits(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(pb{zero(val_type), Int32, UInt16, val_type}(Int32(0)))), fiber(_A))
    # println("Copy to pb success")
    # println("A.lvl: $(A.lvl)")
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j y[i] += A[i, j] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    # time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j (@sieve (Av != 0) (y[i] += (Av[] * x[j]))) where (Av[] = A[i,j])) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y

end

function spmv_finch_packbits_col(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(pb{zero(val_type), Int32, UInt16, val_type}(Int32(0)))), fiber(transpose(_A)))
    # println("Copy to pb success")
    # println("_A: $(transpose(_A))")
    # println("A.lvl: $(A.lvl)")
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    # println("About to call kernel...")
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop j i y[i] += A[j, i] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    # time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch  @loop j i (@sieve (Av != 0) (y[i] += (Av[] * x[j]))) where (Av[] = A[j,i])) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_packbits_mod(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(pb{zero(val_type), Int32, UInt16, val_type}(Int32(0)))), fiber(_A))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    prgm = @Finch.finch_program_instance @loop i j y[i] += A[i, j] * x[j]
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; prgm = $prgm; val_type = $val_type; run_row_pb_spmv(prgm, val_type)) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y

end

function spmv_finch_packbits_col_mod(_A, x, val_type)
    global cold_cache
    A = copyto!(@fiber(d{MyInt}(pb{zero(val_type), Int32, UInt16, val_type}(Int32(0)))), fiber(transpose(_A)))
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    prgm = @Finch.finch_program_instance @loop j i y[i] += A[j, i] * x[j]
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; prgm = $prgm; val_type = $val_type; run_col_pb_spmv(prgm, val_type)) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_repeat_vbl(_A, x, val_type)
    global cold_cache
    # A = dropdefaults!(@fiber(d{MyInt}(sv{MyInt, MyInt}(e(zero(val_type))))), fiber(_A))
    A = to_fiber_mtx_rvb(val_type.(_A), MyInt, MyInt)
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j y[i] += A[i, j] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    # time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop i j (@sieve (Av != 0) (y[i] += (Av[] * x[j]))) where (Av[] = A[i,j])) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function spmv_finch_repeat_vbl_col(_A, x, val_type)
    global cold_cache
    # A = dropdefaults!(@fiber(d{MyInt}(sv{MyInt, MyInt}(e(zero(val_type))))), fiber(transpose(_A)))
    A = to_fiber_mtx_rvb(val_type.(transpose(_A)), MyInt, MyInt)
    x = copyto!(@fiber(d{MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    Av = Scalar(zero(val_type))
    time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch @loop j i y[i] += A[j, i] * x[j]) setup=(clear_cache(cold_cache)) evals=1
    # time = @belapsed (A = $A; x = $x; y = $y; Av = $Av; @finch  @loop j i (@sieve (Av != 0) (y[i] += (Av[] * x[j]))) where (Av[] = A[j,i])) setup=(clear_cache(cold_cache)) evals=1
    size = sizeof1(A)
    return time, size, y
end

function main(result_file)
    global cold_cache
    open(result_file,"w") do f
        println(f, "[")
    end
    comma = false

    for (mtx, key, val_type, col_oriented) in [
        # ("census", "census", Int32, true),
        # ("covtype", "covtype", Int32, true),
        # ("poker", "poker", Int32, true),
        ("humansketches", "humansketches", UInt8, false),
        ("omniglot_train", "omniglot_train", UInt8, false),
        # ("mnist_train", "mnist", UInt8, false),
        # ("emnist_train", "emnist", UInt8, false),
        # ("spgemm", "spgemm", Float64, true),
        # ("kddcup", "kddcup", Float64, true),
        # ("power", "power", Float64, true),
    ]
        A = matrixdepot(mtx)

        if key == "humansketches"
            A = reshape(A[1:5000, :, :], 5000, size(A, 2)*size(A,3))
            A = copy(rawview(channelview(A)))
        elseif key == "omniglot_train"
            A = permutedims(A, (3, 1, 2))
            A = reshape(A[1:10000, :, :], 10000, size(A, 2)*size(A,3))
        elseif key == "mnist" || key == "emnist"
            A = permutedims(A, (3, 1, 2))
            A = reshape(A, size(A, 1), size(A, 2)*size(A,3))
        end

        # A = (A[1:3, 1:3])
        (m, n) = size(A)
        A_nnz = 0 # nnz(SparseMatrixCSC(A))
        println((key, m, n, A_nnz, A_nnz/(m*n)))
        # println((key, m, n, nnz(A)))
        x = rand(n)

        row_methods = [
            # ("taco_sparse", (A, x) -> spmv_taco(A, x, key)),
            ("finch_dense", spmv_finch_dense),
            ("finch_sparse", spmv_finch),
            ("finch_vbl", spmv_finch_vbl),
            ("finch_rl", spmv_finch_rl),
            ("finch_rvbl", spmv_finch_repeat_vbl),
            ("finch_pb", spmv_finch_packbits),
            # ("finch_pb_mod", spmv_finch_packbits_mod),
        ]

        col_methods = [
            ("finch_dense_col", spmv_finch_dense_col),
            ("finch_sparse_col", spmv_finch_col),
            ("finch_vbl_col", spmv_finch_vbl_col),
            ("finch_rl_col", spmv_finch_rl_col),
            ("finch_rvbl_col", spmv_finch_repeat_vbl_col),
            ("finch_pb_col", spmv_finch_packbits_col),
            # ("finch_pb_col_mod", spmv_finch_packbits_col_mod),
        ]

        methods = col_oriented ? col_methods : row_methods

        for (mtd, timer) in methods
            time, A_sz, y_result = timer(A, x, val_type)
            open(result_file,"a") do f
                if comma
                    println(f, ",")
                end
                print(f, """
                    {
                        "matrix": $(repr(mtx)),
                        "m": $(size(A, 1)),
                        "n": $(size(A, 2)),
                        "nnz": $A_nnz,
                        "method": $(repr(mtd)),
                        "val_type": "$(repr(val_type))",
                        "cold_cache": $cold_cache,
                        "col_oriented": $col_oriented,
                        "size": $A_sz,
                        "time": $time
                    }""")
            end
            @info "spmv" mtx size(A, 1) size(A, 2) A_nnz mtd val_type cold_cache col_oriented A_sz time  
            comma = true
        end
    end

    open(result_file,"a") do f
        println()
        println(f, "]")
    end
end

main(ARGS...)