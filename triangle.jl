using Finch
using SparseArrays
using BenchmarkTools
using Scratch

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket

tensor_dir = get_scratch!(@__MODULE__, "tensors")

function triangle_taco(A, key)
    b_file = joinpath(mktempdir(prefix="triangle_taco_$(key)"), "b.ttx")
    persist_dir = joinpath(tensor_dir, "triangle_taco_$(key)")
    mkpath(persist_dir)
    b_ref_file = joinpath(persist_dir, "b_ref.ttx")
    A1_file = joinpath(persist_dir, "A1.ttx")
    A2_file = joinpath(persist_dir, "A2.ttx")
    A3_file = joinpath(persist_dir, "A3.ttx")

    ttwrite(b_file, (), [0], ())
    if !(isfile(A1_file) && isfile(A2_file) && isfile(A3_file))
        (I, J, V) = findnz(A)
        ttwrite(A1_file, (I, J), ones(Int32, length(V)), size(A))
        ttwrite(A2_file, (I, J), ones(Int32, length(V)), size(A))
        ttwrite(A3_file, (I, J), ones(Int32, length(V)), size(A))
    end

    b_ref = Scalar{0.0}()
    A_ref = pattern!(fiber(A))
    @finch @loop i j k b_ref[] += A_ref[i, j] && A_ref[j, k] && A_ref[i, k]

    io = IOBuffer()

    run(pipeline(`./triangle_taco $b_file $A1_file $A2_file $A3_file`, stdout=io))

    b = ttread(b_file)[2][1]

    @assert Float64(b) ≈ Float64(b_ref())

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function triangle_finch_kernel(A)
    c = Scalar{0.0}()
    #println(@finch_code @loop i j k c[] += A[i, j] && A[j, k] && A[i, k])
    @finch @loop i j k c[] += A[i, j] && A[j, k] && A[i, k]
    return c()
end
function triangle_finch(_A, key)
    A = pattern!(fiber(_A))
    return @belapsed triangle_finch_kernel($A)
end

function triangle_finch_gallop_kernel(A)
    c = Scalar{0.0}()
    @finch @loop i j k c[] += A[i, j] && A[j, k::gallop] && A[i, k::gallop]
    return c()
end
function triangle_finch_gallop(_A, key)
    A = pattern!(fiber(_A))
    b_ref = Scalar{0.0}()
    A_ref = pattern!(fiber(A))
    @finch @loop i j k b_ref[] += A_ref[i, j] && A_ref[j, k] && A_ref[i, k]
    b = triangle_finch_gallop_kernel(A)
    println(b, b_ref)
    @assert b_ref() == b
    return @belapsed triangle_finch_gallop_kernel($A)

end

@inline undefs(T::Type, dims::Vararg{Any, N}) where {N} = Array{T, N}(undef, dims...)
function symrcm(A::SparseMatrixCSC{Tv, Ti}) where {Tv, Ti}
    @inbounds begin
        m, n = size(A)

        deg = undefs(Ti, max(m + 2, n))
        bkt = undefs(Ti, n)
        ord = undefs(Ti, n)
        deg[1] = 1
        for i = 2:m + 2
            deg[i] = 0
        end
        for j = 1:n
            deg[A.colptr[j + 1] - A.colptr[j] + 2] += 1
        end
        for i = 1:m + 1
            deg[i + 1] += deg[i]
        end
        for j = 1:n
            d = A.colptr[j + 1] - A.colptr[j] + 1
            k = deg[d]
            ord[k] = j
            deg[d] = k + 1
        end

        k_start = 1

        prm = undefs(Ti, n)
        vst = falses(n)
        k_current = 0
        k_frontier = k_current
        while k_frontier < n
            k_current += 1
            if k_current > k_frontier
                while vst[ord[k_start]]
                    k_start += 1
                end
                j = ord[k_start]
                prm[k_current] = j
                k_frontier = k_current
                vst[j] = true
            else
                j = prm[k_current]
            end
            k_frontier′ = k_frontier
            for q = A.colptr[j]:A.colptr[j + 1] - 1
                i = A.rowval[q]
                if !vst[i]
                    k_frontier′ += 1
                    prm[k_frontier′] = i
                    vst[i] = true
                end
            end

            if k_frontier′ > k_frontier
                let i = prm[k_frontier′]
                    deg[k_frontier′] = A.colptr[i + 1] - A.colptr[i]
                    bkt[k_frontier′] = k_frontier′
                end
                for k = k_frontier′-1:-1:k_frontier + 1
                    i = prm[k]
                    d = A.colptr[i + 1] - A.colptr[i]
                    while k != k_frontier′ && d > deg[k + 1]
                        deg[k] = deg[bkt[k + 1]]
                        prm[k] = prm[bkt[k + 1]]
                        bkt[k] = bkt[k + 1] - (d != deg[k + 1])
                        k = bkt[k + 1]
                    end
                    prm[k] = i
                    deg[k] = d
                    bkt[k] = k
                end

                k_frontier = k_frontier′
            end
        end
        return prm
    end
end

function main()
    for (mtx, key) in [
        ("SNAP/web-NotreDame", "web-NotreDame"),
        ("SNAP/roadNet-PA", "roadNet-PA"),
        ("DIMACS10/sd2010", "sd2010"),
        ("SNAP/soc-Epinions1", "soc-Epinions1"),
        ("SNAP/email-EuAll", "email-EuAll"),
        ("SNAP/wiki-Talk", "wiki-Talk"),
        ("SNAP/web-BerkStan", "web-BerkStan"),
        ("Gleich/flickr", "flickr"),
        ("Gleich/usroads", "usroads"),
        ("Pajek/USpowerGrid", "USpowerGrid"),
    ]
        println(key)
        A = matrixdepot(mtx)
        @info key size(A) nnz(A)

        println("finch_gallop_time: ", triangle_finch_gallop(A, key))
        println("taco_time: ", triangle_taco(A, key))
        println("finch_time: ", triangle_finch(A, key))

        prm = symrcm(A)
        sym_A = A[prm, prm]

        println("sym-$key")

        println("taco_time: ", triangle_taco(sym_A, "sym-$key"))
        println("finch_time: ", triangle_finch(sym_A, "sym-$key"))
        println("finch_gallop_time: ", triangle_finch_gallop(sym_A, "sym-$key"))
    end
end

main()