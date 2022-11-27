using Finch
using SparseArrays
using BenchmarkTools
using Scratch
using Random
using JSON

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket

MatrixDepot.downloadcommand(url::AbstractString, filename::AbstractString="-") =
    `sh -c 'curl -k "'$url'" -Lso "'$filename'"'`

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

boeing = [
("Boeing/bcsstm38", "bcsstm38"),
("Boeing/bcsstm37", "bcsstm37"),
("Boeing/bcsstm35", "bcsstm35"),
("Boeing/bcsstk34", "bcsstk34"),
("Boeing/bcsstm34", "bcsstm34"),
("Boeing/msc01050", "msc01050"),
("Boeing/msc00726", "msc00726"),
("Boeing/nasa1824", "nasa1824"),
("Boeing/msc01440", "msc01440"),
("Boeing/bcsstm39", "bcsstm39"),
("Boeing/msc04515", "msc04515"),
("Boeing/nasa4704", "nasa4704"),
("Boeing/crystm01", "crystm01"),
("Boeing/nasa2910", "nasa2910"),
("Boeing/crystk01", "crystk01"),
("Boeing/bcsstm36", "bcsstm36"),
("Boeing/crystm02", "crystm02"),
("Boeing/bcsstk38", "bcsstk38"),
("Boeing/crystm03", "crystm03"),
("Boeing/crystk02", "crystk02"),
("Boeing/pcrystk02", "pcrystk02"),
("Boeing/bcsstk37", "bcsstk37"),
("Boeing/bcsstk36", "bcsstk36"),
("Boeing/msc23052", "msc23052"),
("Boeing/msc10848", "msc10848"),
("Boeing/bcsstk35", "bcsstk35"),
("Boeing/crystk03", "crystk03"),
("Boeing/pcrystk03", "pcrystk03"),
("Boeing/bcsstk39", "bcsstk39"),
("Boeing/ct20stif", "ct20stif"),
("Boeing/pct20stif", "pct20stif"),
("Boeing/pwtk", "pwtk")
]

hb = [
    ("HB/bcsstm01", "bcsstm01"),
    ("HB/jgl009", "jgl009"),
    ("HB/bcsstm02", "bcsstm02"),
    ("HB/jgl011", "jgl011"),
    ("HB/rgg010", "rgg010"),
    ("HB/bcsstm03", "bcsstm03"),
    ("HB/ibm32", "ibm32"),
    ("HB/bcspwr01", "bcspwr01"),
    ("HB/bcsstm04", "bcsstm04"),
    ("HB/bcsstm22", "bcsstm22"),
    ("HB/bcsstm05", "bcsstm05"),
    ("HB/can_24", "can_24"),
    ("HB/bcspwr02", "bcspwr02"),
    ("HB/lap_25", "lap_25"),
    ("HB/pores_1", "pores_1"),
    ("HB/can_62", "can_62"),
    ("HB/dwt_72", "dwt_72"),
    ("HB/dwt_59", "dwt_59"),
    ("HB/will57", "will57"),
    ("HB/curtis54", "curtis54"),
    ("HB/west0067", "west0067"),
    ("HB/impcol_b", "impcol_b"),
    ("HB/dwt_66", "dwt_66"),
    ("HB/west0156", "west0156"),
    ("HB/can_73", "can_73"),
    ("HB/bcsstk01", "bcsstk01"),
    ("HB/impcol_c", "impcol_c"),
    ("HB/west0132", "west0132"),
    ("HB/bcsstm06", "bcsstm06"),
    ("HB/gre_115", "gre_115"),
    ("HB/ash219", "ash219"),
    ("HB/bcspwr03", "bcspwr03"),
    ("HB/bcsstm20", "bcsstm20"),
    ("HB/west0167", "west0167"),
    ("HB/ash85", "ash85"),
    ("HB/lns_131", "lns_131"),
    ("HB/lnsp_131", "lnsp_131"),
    ("HB/dwt_87", "dwt_87"),
    ("HB/can_61", "can_61"),
    ("HB/impcol_a", "impcol_a"),
    ("HB/nos4", "nos4"),
    ("HB/bcsstk03", "bcsstk03"),
    ("HB/gent113", "gent113"),
    ("HB/ash331", "ash331"),
    ("HB/bcsstk22", "bcsstk22"),
    ("HB/will199", "will199"),
    ("HB/can_96", "can_96"),
    ("HB/bcsstm19", "bcsstm19"),
    ("HB/dwt_234", "dwt_234"),
    ("HB/gre_216a", "gre_216a"),
    ("HB/gre_216b", "gre_216b"),
    ("HB/steam3", "steam3"),
    ("HB/gre_185", "gre_185"),
    ("HB/nos1", "nos1"),
    ("HB/fs_183_1", "fs_183_1"),
    ("HB/fs_183_3", "fs_183_3"),
    ("HB/fs_183_4", "fs_183_4"),
    ("HB/fs_183_6", "fs_183_6"),
    ("HB/bcsstm08", "bcsstm08"),
    ("HB/bcsstm09", "bcsstm09"),
    ("HB/saylr1", "saylr1"),
    ("HB/dwt_162", "dwt_162"),
    ("HB/ash608", "ash608"),
    ("HB/arc130", "arc130"),
    ("HB/can_144", "can_144"),
    ("HB/impcol_e", "impcol_e"),
    ("HB/impcol_d", "impcol_d"),
    ("HB/can_161", "can_161"),
    ("HB/dwt_198", "dwt_198"),
    ("HB/gre_343", "gre_343"),
    ("HB/dwt_245", "dwt_245"),
    ("HB/bcsstm11", "bcsstm11"),
    ("HB/can_187", "can_187"),
    ("HB/nnc261", "nnc261"),
    ("HB/abb313", "abb313"),
    ("HB/bcspwr04", "bcspwr04"),
    ("HB/bcspwr05", "bcspwr05"),
    ("HB/dwt_221", "dwt_221"),
    ("HB/494_bus", "494_bus"),
    ("HB/shl_0", "shl_0"),
    ("HB/shl_400", "shl_400"),
    ("HB/shl_200", "shl_200"),
    ("HB/west0497", "west0497"),
    ("HB/dwt_209", "dwt_209"),
    ("HB/lshp_265", "lshp_265"),
    ("HB/plskz362", "plskz362"),
    ("HB/can_229", "can_229"),
    ("HB/west0479", "west0479"),
    ("HB/ash958", "ash958"),
    ("HB/bcsstm26", "bcsstm26"),
    ("HB/west0381", "west0381"),
    ("HB/gre_512", "gre_512"),
    ("HB/ash292", "ash292"),
    ("HB/bcsstk05", "bcsstk05"),
    ("HB/lund_b", "lund_b"),
    ("HB/dwt_310", "dwt_310"),
    ("HB/lund_a", "lund_a"),
    ("HB/str_0", "str_0"),
    ("HB/662_bus", "662_bus"),
    ("HB/dwt_307", "dwt_307"),
    ("HB/can_292", "can_292"),
    ("HB/fs_680_1", "fs_680_1"),
    ("HB/fs_680_2", "fs_680_2"),
    ("HB/fs_680_3", "fs_680_3"),
    ("HB/mcca", "mcca"),
    ("HB/lshp_406", "lshp_406"),
    ("HB/lns_511", "lns_511"),
    ("HB/lnsp_511", "lnsp_511"),
    ("HB/west0655", "west0655"),
    ("HB/wm1", "wm1"),
    ("HB/can_256", "can_256"),
    ("HB/wm2", "wm2"),
    ("HB/wm3", "wm3"),
    ("HB/dwt_361", "dwt_361"),
    ("HB/str_200", "str_200"),
    ("HB/can_268", "can_268"),
    ("HB/bcsstm23", "bcsstm23"),
    ("HB/bcsstk20", "bcsstk20"),
    ("HB/dwt_492", "dwt_492"),
    ("HB/str_400", "str_400"),
    ("HB/dwt_346", "dwt_346"),
    ("HB/685_bus", "685_bus"),
    ("HB/nos6", "nos6"),
    ("HB/bp_0", "bp_0"),
    ("HB/str_600", "str_600"),
    ("HB/pores_3", "pores_3"),
    ("HB/dwt_193", "dwt_193"),
    ("HB/dwt_512", "dwt_512"),
    ("HB/west0989", "west0989"),
    ("HB/bcsstm24", "bcsstm24"),
    ("HB/dwt_419", "dwt_419"),
    ("HB/bcsstm21", "bcsstm21"),
    ("HB/bcsstk04", "bcsstk04"),
    ("HB/saylr3", "saylr3"),
    ("HB/sherman1", "sherman1"),
    ("HB/steam1", "steam1"),
    ("HB/sherman4", "sherman4"),
    ("HB/bp_200", "bp_200"),
    ("HB/can_445", "can_445"),
    ("HB/lshp_577", "lshp_577"),
    ("HB/young3c", "young3c"),
    ("HB/bp_400", "bp_400"),
    ("HB/nnc666", "nnc666"),
    ("HB/1138_bus", "1138_bus"),
    ("HB/young1c", "young1c"),
    ("HB/young2c", "young2c"),
    ("HB/young4c", "young4c"),
    ("HB/nos2", "nos2"),
    ("HB/bp_600", "bp_600"),
    ("HB/fs_541_1", "fs_541_1"),
    ("HB/fs_541_2", "fs_541_2"),
    ("HB/fs_541_3", "fs_541_3"),
    ("HB/fs_541_4", "fs_541_4"),
    ("HB/bcsstk02", "bcsstk02"),
    ("HB/bp_800", "bp_800"),
    ("HB/nos7", "nos7"),
    ("HB/bp_1000", "bp_1000"),
    ("HB/hor_131", "hor_131"),
    ("HB/bp_1200", "bp_1200"),
    ("HB/illc1033", "illc1033"),
    ("HB/well1033", "well1033"),
    ("HB/bp_1400", "bp_1400"),
    ("HB/bp_1600", "bp_1600"),
    ("HB/dwt_592", "dwt_592"),
    ("HB/dwt_607", "dwt_607"),
    ("HB/nos5", "nos5"),
    ("HB/lshp_778", "lshp_778"),
    ("HB/bcspwr06", "bcspwr06"),
    ("HB/west1505", "west1505"),
    ("HB/gre_1107", "gre_1107"),
    ("HB/plat362", "plat362"),
    ("HB/bcspwr07", "bcspwr07"),
    ("HB/orsirr_2", "orsirr_2"),
    ("HB/fs_760_1", "fs_760_1"),
    ("HB/fs_760_2", "fs_760_2"),
    ("HB/fs_760_3", "fs_760_3"),
    ("HB/dwt_758", "dwt_758"),
    ("HB/dwt_503", "dwt_503"),
    ("HB/jpwh_991", "jpwh_991"),
    ("HB/bcspwr08", "bcspwr08"),
    ("HB/jagmesh1", "jagmesh1"),
    ("HB/bcspwr09", "bcspwr09"),
    ("HB/can_715", "can_715"),
    ("HB/bcsstk19", "bcsstk19"),
    ("HB/orsirr_1", "orsirr_1"),
    ("HB/jagmesh2", "jagmesh2"),
    ("HB/lshp1009", "lshp1009"),
    ("HB/can_634", "can_634"),
    ("HB/bcsstm07", "bcsstm07"),
    ("HB/dwt_869", "dwt_869"),
    ("HB/west2021", "west2021"),
    ("HB/jagmesh3", "jagmesh3"),
    ("HB/dwt_918", "dwt_918"),
    ("HB/dwt_878", "dwt_878"),
    ("HB/jagmesh7", "jagmesh7"),
    ("HB/jagmesh8", "jagmesh8"),
    ("HB/mahindas", "mahindas"),
    ("HB/gr_30_30", "gr_30_30"),
    ("HB/jagmesh5", "jagmesh5"),
    ("HB/bcsstk06", "bcsstk06"),
    ("HB/bcsstk07", "bcsstk07"),
    ("HB/dwt_1007", "dwt_1007"),
    ("HB/nnc1374", "nnc1374"),
    ("HB/dwt_1005", "dwt_1005"),
    ("HB/lshp1270", "lshp1270"),
    ("HB/illc1850", "illc1850"),
    ("HB/well1850", "well1850"),
    ("HB/jagmesh6", "jagmesh6"),
    ("HB/jagmesh9", "jagmesh9"),
    ("HB/jagmesh4", "jagmesh4"),
    ("HB/pores_2", "pores_2"),
    ("HB/plsk1919", "plsk1919"),
    ("HB/can_838", "can_838"),
    ("HB/dwt_1242", "dwt_1242"),
    ("HB/lshp1561", "lshp1561"),
    ("HB/watt_1", "watt_1"),
    ("HB/watt_2", "watt_2"),
    ("HB/can_1054", "can_1054"),
    ("HB/can_1072", "can_1072"),
    ("HB/lshp1882", "lshp1882"),
    ("HB/bcsstk08", "bcsstk08"),
    ("HB/steam2", "steam2"),
    ("HB/orsreg_1", "orsreg_1"),
    ("HB/blckhole", "blckhole"),
    ("HB/lshp2233", "lshp2233"),
    ("HB/bcsstm25", "bcsstm25"),
    ("HB/nos3", "nos3"),
    ("HB/dwt_992", "dwt_992"),
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
    ("HB/lock_700", "lock_700"),
    ("HB/saylr4", "saylr4"),
    ("HB/sstmodel", "sstmodel"),
    ("HB/sherman2", "sherman2"),
    ("HB/lshp3466", "lshp3466"),
    ("HB/mcfe", "mcfe"),
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
    ("HB/mbeause", "mbeause"),
    ("HB/beause", "beause"),
    ("HB/bcsstk23", "bcsstk23"),
    ("HB/gemat1", "gemat1"),
    ("HB/mbeacxc", "mbeacxc"),
    ("HB/mbeaflw", "mbeaflw"),
    ("HB/beacxc", "beacxc"),
    ("HB/lock1074", "lock1074"),
    ("HB/beaflw", "beaflw"),
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

function main(result_file)
    open(result_file,"w") do f
        println(f, "[")
    end
    comma = false
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
        println(f, "]")
    end
end



main(ARGS...)
