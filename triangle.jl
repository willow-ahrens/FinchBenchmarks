using Finch
using SparseArrays
using BenchmarkTools
using Scratch
using Profile
using JSON

using MatrixDepot
using TensorMarket

const MyInt=Int32

MatrixDepot.downloadcommand(url::AbstractString, filename::AbstractString="-") =
    `sh -c 'curl -k "'$url'" -Lso "'$filename'"'`

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

function triangle_taco_sparse(A, key)
    c_file = joinpath(mktempdir(prefix="triangle_taco_sparse_$(key)"), "c.ttx")
    persist_dir = joinpath(get_scratch!("Finch-CGO-2023"), "triangle_taco_sparse_$(key)")
    mkpath(persist_dir)
    A_file = joinpath(persist_dir, "A.ttx")
    A2_file = joinpath(persist_dir, "A2.ttx")
    AT_file = joinpath(persist_dir, "AT.ttx")

    ttwrite(c_file, (), [0], ())
    if !(isfile(A_file) && isfile(A2_file) && isfile(AT_file))
        (I, J, V) = findnz(A)
        ttwrite(A_file, (I, J), ones(Int32, length(V)), size(A))
        ttwrite(A2_file, (I, J), ones(Int32, length(V)), size(A))
        ttwrite(AT_file, (J, I), ones(Int32, length(V)), size(A))
    end

    io = IOBuffer()

    withenv("DYLD_FALLBACK_LIBRARY_PATH"=>"./taco/build/lib", "LD_LIBRARY_PATH" => "./taco/build/lib", "TACO_CFLAGS" => "-O3 -ffast-math -std=c99 -march=native -ggdb") do
        run(pipeline(`./triangle_taco $c_file $A_file $A2_file $AT_file`, stdout=io))
    end

    c = ttread(c_file)[2][1]

    return (parse(Int64, String(take!(io))) * 1.0e-9, c)
end

function triangle_finch_kernel(A, AT)
    c = Scalar{0}()
    @finch @loop i j k c[] += A[i, j] && A[j, k] && AT[i, k]
    return c()
end
function triangle_finch_sparse(_A, key)
    A = pattern!(copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(_A)))
    AT = pattern!(copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(permutedims(_A))))
    time = @belapsed triangle_finch_kernel($A, $AT) setup=(clear_cache())
    c = triangle_finch_kernel(A, AT)
    return time, c
end

function triangle_finch_gallop_kernel(A, AT)
    c = Scalar{0}()
    @finch @loop i j k c[] += A[i, j] && A[j, k::gallop] && AT[i, k::gallop]
    return c()
end
function triangle_finch_gallop(_A, key)
    A = pattern!(copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(_A)))
    AT = pattern!(copyto!(@fiber(d{MyInt}(sl{MyInt, MyInt}(e(0.0)))), fiber(permutedims(_A))))
    time = @belapsed triangle_finch_gallop_kernel($A, $AT) setup=(clear_cache())
    c = triangle_finch_gallop_kernel(A, AT)
    return (time, c)
end

snap_short = [
    ("SNAP/email-Eu-core", "email-Eu-core"),
    ("SNAP/email-Eu-core-temporal", "email-Eu-core-temporal"),
    ("SNAP/CollegeMsg", "CollegeMsg"),
    ("SNAP/soc-sign-bitcoin-alpha", "soc-sign-bitcoin-alpha"),
    ("SNAP/ca-GrQc", "ca-GrQc"),
    ("SNAP/com-Youtube", "com-Youtube"),
    ("SNAP/as-caida", "as-caida"),
    ("SNAP/Oregon-1", "Oregon-1"),
    ("SNAP/as-735", "as-735"),
    ("SNAP/loc-Gowalla", "loc-Gowalla"),
] 

snap = [
    ("SNAP/email-Eu-core", "email-Eu-core"),
    ("SNAP/email-Eu-core-temporal", "email-Eu-core-temporal"),
    ("SNAP/CollegeMsg", "CollegeMsg"),
    ("SNAP/soc-sign-bitcoin-alpha", "soc-sign-bitcoin-alpha"),
    ("SNAP/ca-GrQc", "ca-GrQc"),
    ("SNAP/soc-sign-bitcoin-otc", "soc-sign-bitcoin-otc"),
    ("SNAP/p2p-Gnutella08", "p2p-Gnutella08"),
    ("SNAP/as-735", "as-735"),
    ("SNAP/p2p-Gnutella09", "p2p-Gnutella09"),
    ("SNAP/wiki-Vote", "wiki-Vote"),
    ("SNAP/p2p-Gnutella06", "p2p-Gnutella06"),
    ("SNAP/p2p-Gnutella05", "p2p-Gnutella05"),
    ("SNAP/ca-HepTh", "ca-HepTh"),
    ("SNAP/p2p-Gnutella04", "p2p-Gnutella04"),
    ("SNAP/wiki-RfA", "wiki-RfA"),
    ("SNAP/Oregon-1", "Oregon-1"),
    ("SNAP/Oregon-2", "Oregon-2"),
    ("SNAP/ca-HepPh", "ca-HepPh"),
    ("SNAP/ca-AstroPh", "ca-AstroPh"),
    ("SNAP/p2p-Gnutella25", "p2p-Gnutella25"),
    ("SNAP/ca-CondMat", "ca-CondMat"),
    ("SNAP/sx-mathoverflow", "sx-mathoverflow"),
    ("SNAP/p2p-Gnutella24", "p2p-Gnutella24"),
    ("SNAP/cit-HepTh", "cit-HepTh"),
    ("SNAP/as-caida", "as-caida"),
    ("SNAP/cit-HepPh", "cit-HepPh"),
    ("SNAP/p2p-Gnutella30", "p2p-Gnutella30"),
    ("SNAP/email-Enron", "email-Enron"),
    ("SNAP/loc-Brightkite", "loc-Brightkite"),
    ("SNAP/p2p-Gnutella31", "p2p-Gnutella31"),
    ("SNAP/soc-Epinions1", "soc-Epinions1"),
    ("SNAP/soc-sign-Slashdot081106", "soc-sign-Slashdot081106"),
    ("SNAP/soc-Slashdot0811", "soc-Slashdot0811"),
    ("SNAP/soc-sign-Slashdot090216", "soc-sign-Slashdot090216"),
    ("SNAP/soc-sign-Slashdot090221", "soc-sign-Slashdot090221"),
    ("SNAP/soc-Slashdot0902", "soc-Slashdot0902"),
    ("SNAP/soc-sign-epinions", "soc-sign-epinions"),
    ("SNAP/sx-askubuntu", "sx-askubuntu"),
    ("SNAP/sx-superuser", "sx-superuser"),
    ("SNAP/loc-Gowalla", "loc-Gowalla"),
    ("SNAP/amazon0302", "amazon0302"),
    ("SNAP/email-EuAll", "email-EuAll"),
    ("SNAP/web-Stanford", "web-Stanford"),
    ("SNAP/com-DBLP", "com-DBLP"),
    ("SNAP/web-NotreDame", "web-NotreDame"),
    ("SNAP/com-Amazon", "com-Amazon"),
    ("SNAP/amazon0312", "amazon0312"),
    ("SNAP/amazon0601", "amazon0601"),
    ("SNAP/amazon0505", "amazon0505"),
    ("SNAP/higgs-twitter", "higgs-twitter"),
    ("SNAP/web-BerkStan", "web-BerkStan"),
    ("SNAP/web-Google", "web-Google"),
    ("SNAP/roadNet-PA", "roadNet-PA"),
    ("SNAP/com-Youtube", "com-Youtube"),
    ("SNAP/wiki-talk-temporal", "wiki-talk-temporal"),
    ("SNAP/roadNet-TX", "roadNet-TX"),
    ("SNAP/soc-Pokec", "soc-Pokec"),
]

function main(result_file, short="long")
    global snap
    open(result_file,"w") do f
        println(f, "[")
    end
    comma = false

    if short == "short"
        snap = snap_short
    end

    for (mtx, key) in snap
        A = SparseMatrixCSC(matrixdepot(mtx))
        println((key, size(A), nnz(A)))

        time, ref = triangle_taco_sparse(A, key)

        for (method, f) in [
            ("taco_sparse", triangle_taco_sparse),
            ("finch_sparse", triangle_finch_sparse),
            ("finch_gallop", triangle_finch_gallop),
        ]

            time, res = f(A, key)
            @assert res == ref

            open(result_file,"a") do f
                if comma
                    println(f, ",")
                end
                print(f, """
                    {
                        "matrix": $(repr(mtx)),
                        "n": $(size(A, 1)),
                        "nnz": $(nnz(A)),
                        "method": $(repr(method)),
                        "time": $time
                    }""")
            end
            @info "triangle" mtx size(A, 1) nnz(A) method time
            comma = true
        end
    end
    open(result_file,"a") do f
        println()
        println(f, "]")
    end
end

main(ARGS...)