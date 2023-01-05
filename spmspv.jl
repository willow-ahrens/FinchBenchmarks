using Finch
using SparseArrays
using BenchmarkTools
using Scratch
using Random
using JSON

using MatrixDepot
using TensorMarket

const MyInt = Int

function MatrixDepot.downloadcommand(url::AbstractString, filename::AbstractString="-")
    `sh -c 'curl -k "'$url'" -Lso "'$filename'"'`
end

MatrixDepot.init()

global dummySize=5000000
global dummyA=[]
global dummyB=[]

@noinline
function clear_cache()
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
    if !(isfile(A_file))
        ((I, J), V) = ffindnz(A)
        ttwrite(A_file, (I, J), V, size(_A))
    end

    io = IOBuffer()

    println("running")

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./taco/build/lib", "LD_LIBRARY_PATH" => "./taco/build/lib", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        run(pipeline(`./spmspv_taco $y_file $A_file $x_file`, stdout=io))
    end

    y = fsparse(ttread(y_file)...)

    @finch @loop i j y_ref[i] += A[i, j] * x[j]

    #println((FiberArray(y)[1:10], FiberArray(y_ref)[1:10]))
    #@assert FiberArray(y) ≈ FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmspv_finch(_A, x)
    A = copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(_A))
    x = copyto!(@fiber(sl{MyInt, MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j]) setup=(clear_cache()) evals=1
end

function spmspv_gallop_finch(_A, x)
    A = copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(_A))
    x = copyto!(@fiber(sl{MyInt, MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j::gallop] * x[j::gallop]) setup=(clear_cache()) evals=1
end

function spmspv_lead_finch(_A, x)
    A = copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(_A))
    x = copyto!(@fiber(sl{MyInt, MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j::gallop] * x[j]) setup=(clear_cache()) evals=1
end

function spmspv_follow_finch(_A, x)
    A = copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(_A))
    x = copyto!(@fiber(sl{MyInt, MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j::gallop]) setup=(clear_cache()) evals=1
end

function spmspv_finch_vbl(_A, x)
    A = copyto!(@fiber(d{MyInt}(sv{MyInt, MyInt}(e(0.0)))), fiber(_A))
    x = copyto!(@fiber(sl{MyInt, MyInt}(e(0.0))), x)
    y = @fiber(d{MyInt}(e(0.0)))
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j::gallop] * x[j]) setup=(clear_cache()) evals=1
end

hb_short = [
    ("HB/bcsstm08", "bcsstm08"),
    ("HB/bcsstm09", "bcsstm09"),
    ("HB/bcsstm11", "bcsstm11"),
    ("HB/bcsstm26", "bcsstm26"),
    ("HB/bcsstm23", "bcsstm23"),
    ("HB/bcsstm25", "bcsstm25"),
    ("HB/bcsstk32", "bcsstk32"),
    ("HB/cegb2802", "cegb2802"),
    ("HB/bcsstk30", "bcsstk30"),
    ("HB/bcsstk31", "bcsstk31"),
]

hb = [
    ("HB/bcsstm08", "bcsstm08"),
    ("HB/bcsstm09", "bcsstm09"),
    ("HB/bcsstm11", "bcsstm11"),
    ("HB/bcsstm26", "bcsstm26"),
    ("HB/bcsstm23", "bcsstm23"),
    ("HB/bcsstm24", "bcsstm24"),
    ("HB/bcsstm21", "bcsstm21"),
    ("HB/saylr3", "saylr3"),
    ("HB/sherman1", "sherman1"),
    ("HB/sherman4", "sherman4"),
    ("HB/1138_bus", "1138_bus"),
    ("HB/bcspwr06", "bcspwr06"),
    ("HB/west1505", "west1505"),
    ("HB/gre_1107", "gre_1107"),
    ("HB/bcspwr07", "bcspwr07"),
    ("HB/bcspwr08", "bcspwr08"),
    ("HB/bcspwr09", "bcspwr09"),
    ("HB/orsirr_1", "orsirr_1"),
    ("HB/jagmesh2", "jagmesh2"),
    ("HB/lshp1009", "lshp1009"),
    ("HB/west2021", "west2021"),
    ("HB/jagmesh3", "jagmesh3"),
    ("HB/jagmesh7", "jagmesh7"),
    ("HB/jagmesh8", "jagmesh8"),
    ("HB/mahindas", "mahindas"),
    ("HB/jagmesh5", "jagmesh5"),
    ("HB/dwt_1007", "dwt_1007"),
    ("HB/nnc1374", "nnc1374"),
    ("HB/dwt_1005", "dwt_1005"),
    ("HB/lshp1270", "lshp1270"),
    ("HB/jagmesh6", "jagmesh6"),
    ("HB/jagmesh9", "jagmesh9"),
    ("HB/jagmesh4", "jagmesh4"),
    ("HB/pores_2", "pores_2"),
    ("HB/plsk1919", "plsk1919"),
    ("HB/dwt_1242", "dwt_1242"),
    ("HB/lshp1561", "lshp1561"),
    ("HB/watt_1", "watt_1"),
    ("HB/watt_2", "watt_2"),
    ("HB/can_1054", "can_1054"),
    ("HB/can_1072", "can_1072"),
    ("HB/lshp1882", "lshp1882"),
    ("HB/bcsstk08", "bcsstk08"),
    ("HB/orsreg_1", "orsreg_1"),
    ("HB/blckhole", "blckhole"),
    ("HB/lshp2233", "lshp2233"),
    ("HB/bcsstm25", "bcsstm25"),
    ("HB/lshp2614", "lshp2614"),
    ("HB/bcsstk09", "bcsstk09"),
    ("HB/eris1176", "eris1176"),
    ("HB/bcsstm12", "bcsstm12"),
    ("HB/sherman3", "sherman3"),
    ("HB/sherman5", "sherman5"),
    ("HB/lshp3025", "lshp3025"),
    ("HB/bcspwr10", "bcspwr10"),
    ("HB/bcsstm13", "bcsstm13"),
    ("HB/bcsstk10", "bcsstk10"),
    ("HB/bcsstm10", "bcsstm10"),
    ("HB/saylr4", "saylr4"),
    ("HB/sstmodel", "sstmodel"),
    ("HB/sherman2", "sherman2"),
    ("HB/lshp3466", "lshp3466"),
    ("HB/dwt_2680", "dwt_2680"),
    ("HB/lns_3937", "lns_3937"),
    ("HB/lnsp3937", "lnsp3937"),
    ("HB/bcsstk21", "bcsstk21"),
    ("HB/zenios", "zenios"),
    ("HB/bcsstk26", "bcsstk26"),
    ("HB/plat1919", "plat1919"),
    ("HB/gemat12", "gemat12"),
    ("HB/gemat11", "gemat11"),
    ("HB/bcsstk11", "bcsstk11"),
    ("HB/bcsstk12", "bcsstk12"),
    ("HB/bcsstk23", "bcsstk23"),
    ("HB/gemat1", "gemat1"),
    ("HB/lock1074", "lock1074"),
    ("HB/bcsstk27", "bcsstk27"),
    ("HB/bcsstm27", "bcsstm27"),
    ("HB/bcsstk14", "bcsstk14"),
    ("HB/cegb3306", "cegb3306"),
    ("HB/cegb3024", "cegb3024"),
    ("HB/lock2232", "lock2232"),
    ("HB/bcsstk13", "bcsstk13"),
    ("HB/orani678", "orani678"),
    ("HB/bcsstk15", "bcsstk15"),
    ("HB/bcsstk18", "bcsstk18"),
    ("HB/bcsstk24", "bcsstk24"),
    ("HB/lock3491", "lock3491"),
    ("HB/bcsstk28", "bcsstk28"),
    ("HB/man_5976", "man_5976"),
    ("HB/bcsstk25", "bcsstk25"),
    ("HB/cegb2802", "cegb2802"),
    ("HB/bcsstk16", "bcsstk16"),
    ("HB/cegb2919", "cegb2919"),
    ("HB/bcsstk17", "bcsstk17"),
    ("HB/psmigr_2", "psmigr_2"),
    ("HB/psmigr_1", "psmigr_1"),
    ("HB/psmigr_3", "psmigr_3"),
    ("HB/bcsstk33", "bcsstk33"),
    ("HB/bcsstk29", "bcsstk29"),
    ("HB/bcsstk31", "bcsstk31"),
    ("HB/bcsstk32", "bcsstk32"),
    ("HB/bcsstk30", "bcsstk30"),
]

function main(result_file, short="long")
    global hb
    open(result_file,"w") do f
        println(f, "[")
    end
    comma = false

    if short == "short" 
        hb = hb_short
    end

    for (mtx, key) in hb
    	raw = matrixdepot(mtx)
        if !(eltype(raw) <: Real)
            continue
        end
        A = SparseMatrixCSC{Float64}(raw)
        (m, n) = size(A)
        if m < 1000 || n < 1000
            continue
        end
        println("0.1 density: ", (key, m, n, nnz(A)))
        for run = 1:1
            for (x, xname) in [
                (fsprand((n,), 0.1), "0.1 density"),
                (Fiber(SparseList(n, [1, 11], sort(randperm(n)[1:10]), Element{0.0}(rand(10)))), "10 count")
            ]

                for (mtd, timer) in [
                    ("taco_sparse", (A, x) -> spmspv_taco(A, x, key)),
                    ("finch_sparse", spmspv_finch),
                    ("finch_gallop", spmspv_gallop_finch),
                    ("finch_lead", spmspv_lead_finch),
                    ("finch_follow", spmspv_follow_finch),
                    ("finch_vbl", spmspv_finch_vbl),
                ]
                    time = timer(A, x)
                    open(result_file,"a") do f
                        if comma
                            println(f, ",")
                        end
                        print(f, """
                            {
                                "matrix": $(repr(mtx)),
                                "n": $(size(A, 1)),
                                "nnz": $(nnz(A)),
                                "x": $(repr(xname)),
                                "run": $(run),
                                "method": $(repr(mtd)),
                                "time": $time
                            }""")
                    end
                    @info "spmspv" mtx size(A, 1) nnz(A) xname run mtd time
                    comma = true
                end
            end
        end
    end

    open(result_file,"a") do f
        println()
        println(f, "]")
    end
end



main(ARGS...)
