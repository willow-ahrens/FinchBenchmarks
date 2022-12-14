using Finch
using SparseArrays
using BenchmarkTools
using Scratch
using Random
using JSON

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket

# function MatrixDepot.downloadcommand(url::AbstractString, filename::AbstractString="-")
#     `sh -c 'curl -k "'$url'" -Lso "'$filename'"'`
# end

# MatrixDepot.init()

function sddmm_finch(_B, _C, _D)
    B = fiber(_B)
    C = copyto!(@fiber(d(d(e(0.0)))), _C)
    D = copyto!(@fiber(d(d(e(0.0)))), _D)
    
    A = similar(B)
    return @belapsed (A = $A; B = $B; C = $C; D = $D; @finch @loop i j k A[i,j] = B[i, j] * C[i,k] * D[k,j])
end

mtxs = [
    ("Bova/rma10", "rma10"),
    ("Boeing/pwtk", "pwtk"),
    ("Hamm/scircuit", "scircuit"),
    ("Williams/mac_econ_fwd500", "mac_econ_fwd500"),
    ("Williams/cop20k_A", "cop20k_A"),
    ("Williams/cant", "cant"),
]

function main(result_file)
    global mtxs
    open(result_file,"w") do f
        println(f, "[")
    end
    comma = false

    k = 1000

    for (mtx, key) in mtxs
        matrixdepot(mtx)
    end

    for (mtx, key) in mtxs
    	raw = matrixdepot(mtx)
        if !(eltype(raw) <: Real)
            println("real :( $(mtx)")
            continue
        end
        B = SparseMatrixCSC{Float64}(raw)
        (m, n) = size(B)
        if m < 1000 || n < 1000
            println("Not big enough :( $(mtx)")
            continue
        end
        C = rand(m,k)
        D = rand(k,n)
        for (mtd, timer) in [
            # ("taco_sparse", (A, x) -> spmspv_taco(A, x, key)),
            ("finch_sparse", sddmm_finch),
            # ("finch_gallop", spmspv_gallop_finch),
            # ("finch_lead", spmspv_lead_finch),
            # ("finch_follow", spmspv_follow_finch),
            # ("finch_vbl", spmspv_finch_vbl),
        ]
            time = timer(B, C, D)
            open(result_file,"a") do f
                if comma
                    println(f, ",")
                end
                print(f, """
                    {
                        "matrix": $(repr(mtx)),
                        "n": $(size(B, 1)),
                        "nnz": $(nnz(B)),
                        "k": $(k),
                        "method": $(repr(mtd)),
                        "time": $time
                    }""")
            end
            @info "sddmm" mtx size(B, 1) nnz(B) k mtd time
            comma = true
        end
    end

    open(result_file,"a") do f
        println()
        println(f, "]")
    end
end

main(ARGS...)
