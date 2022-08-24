using Finch
using SparseArrays
using BenchmarkTools
using Scratch
using Profile

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

    b_ref = Scalar{0}()
    A_ref = pattern!(fiber(A))
    @finch @loop i j k b_ref[] += A_ref[i, j] && A_ref[j, k] && A_ref[i, k]

    io = IOBuffer()

    run(pipeline(`./triangle_taco $b_file $A1_file $A2_file $A3_file`, stdout=io))

    b = ttread(b_file)[2][1]

    @assert Float64(b) ≈ Float64(b_ref())

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function triangle_finch_kernel(A)
    c = Scalar{0}()
    @finch @loop i j k c[] += A[i, j] && A[j, k] && A[i, k]
    return c()
end
function triangle_finch(_A, key)
    A = copyto!(Fiber(Dense(SparseList{Int32}(Element(0.0)))), fiber(_A))
    A = pattern!(A)
    #return @belapsed triangle_finch_kernel($A)
    foo(A)
    @profile foo(A)
    Profile.print()
    exit()
    return @belapsed foo($A)
end

function triangle_finch_gallop_kernel(A)
    c = Scalar{0}()
    @finch @loop i j k c[] += A[i, j] && A[j, k::gallop] && A[i, k::gallop]
    return c()
end
function triangle_finch_gallop(_A, key)
    A = pattern!(fiber(_A))
    b_ref = Scalar{0}()
    @finch @loop i j k b_ref[] += A[i, j] && A[j, k] && A[i, k]
    b = triangle_finch_gallop_kernel(A)
    @assert b_ref() == b
    return @belapsed triangle_finch_gallop_kernel($A)
end

function main()
    for (mtx, key) in [
        #("SNAP/web-NotreDame", "web-NotreDame"),
        #("SNAP/roadNet-PA", "roadNet-PA"),
        #("DIMACS10/sd2010", "sd2010"),
        #("SNAP/soc-Epinions1", "soc-Epinions1"),
        #("SNAP/email-EuAll", "email-EuAll"),
        #("SNAP/wiki-Talk", "wiki-Talk"),
        ("SNAP/web-BerkStan", "web-BerkStan"),
        #("Gleich/flickr", "flickr"),
        #("Gleich/usroads", "usroads"),
        #("Pajek/USpowerGrid", "USpowerGrid"),
    ]
        println(key)
        A = SparseMatrixCSC(matrixdepot(mtx))
        @info key size(A) nnz(A)
        println(maximum(A.colptr[2:end] - A.colptr[1:end-1]))

        #println("taco_time: ", triangle_taco(A, key))
        println("finch_time: ", triangle_finch(A, key))
        println("finch_gallop_time: ", triangle_finch_gallop(A, key))

    end
end

foo(A) = @inbounds begin
    println("hi")
    A_lvl = A.lvl
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_pos_alloc = length(A_lvl_2.pos)
    A_lvl_2_idx_alloc = length(A_lvl_2.idx)
    A_lvl_3 = A.lvl
    A_lvl_4 = A_lvl_3.lvl
    A_lvl_4_pos_alloc = length(A_lvl_4.pos)
    A_lvl_4_idx_alloc = length(A_lvl_4.idx)
    A_lvl_5 = A.lvl
    A_lvl_6 = A_lvl_5.lvl
    A_lvl_6_pos_alloc = length(A_lvl_6.pos)
    A_lvl_6_idx_alloc = length(A_lvl_6.idx)
    j_stop = A_lvl_2.I
    k_stop = A_lvl_4.I
    i_stop = A_lvl.I
    c_val = 0
    for i = 1:i_stop
        A_lvl_q = (1 - 1) * A_lvl.I + i
        A_lvl_5_q = (1 - 1) * A_lvl_5.I + i
        A_lvl_2_q_start = A_lvl_2.pos[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_2.pos[A_lvl_q + 1]
        if A_lvl_2_q_start < A_lvl_2_q_stop
            A_lvl_2_i_start = A_lvl_2.idx[A_lvl_2_q_start]
            A_lvl_2_i_stop = A_lvl_2.idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i_start = 1
            A_lvl_2_i_stop = 0
        end
        A_lvl_6_q_start = A_lvl_6.pos[A_lvl_5_q]
        A_lvl_6_q_stop = A_lvl_6.pos[A_lvl_5_q + 1]
        if A_lvl_6_q_start < A_lvl_6_q_stop
            A_lvl_6_i_start = A_lvl_6.idx[A_lvl_6_q_start]
            A_lvl_6_i_stop = A_lvl_6.idx[A_lvl_6_q_stop - 1]
        else
            A_lvl_6_i_start = 1
            A_lvl_6_i_stop = 0
        end
        A_lvl_2_q = A_lvl_2_q_start
        A_lvl_2_i = A_lvl_2_i_start
        j = 1
        j_start = j
        phase_start = max(j_start)
        phase_stop = min(A_lvl_2_i_stop, j_stop)
        if phase_stop >= phase_start
            j = j
            j = phase_start
            while A_lvl_2_q < A_lvl_2_q_stop && A_lvl_2.idx[A_lvl_2_q] < phase_start
                A_lvl_2_q += 1
            end
            while j <= phase_stop
                j_start_2 = j
                A_lvl_2_i = A_lvl_2.idx[A_lvl_2_q]
                phase_stop_2 = A_lvl_2_i
                j_2 = j
                if A_lvl_2_i == phase_stop_2
                    j_3 = phase_stop_2
                    A_lvl_3_q = (1 - 1) * A_lvl_3.I + j_3
                    A_lvl_4_q_start = A_lvl_4.pos[A_lvl_3_q]
                    A_lvl_4_q_stop = A_lvl_4.pos[A_lvl_3_q + 1]
                    if A_lvl_4_q_start < A_lvl_4_q_stop
                        A_lvl_4_i_start = A_lvl_4.idx[A_lvl_4_q_start]
                        A_lvl_4_i_stop = A_lvl_4.idx[A_lvl_4_q_stop - 1]
                    else
                        A_lvl_4_i_start = 1
                        A_lvl_4_i_stop = 0
                    end
                    A_lvl_4_q = A_lvl_4_q_start
                    A_lvl_4_i = A_lvl_4_i_start
                    A_lvl_6_q = A_lvl_6_q_start
                    A_lvl_6_i = A_lvl_6_i_start
                    k = 1
                    k_start = k
                    phase_start_3 = max(k_start)
                    phase_stop_3 = min(A_lvl_4_i_stop, A_lvl_6_i_stop, k_stop)
                    if phase_stop_3 >= phase_start_3
                        k = k
                        k = phase_start_3
                        #while A_lvl_4_q < A_lvl_4_q_stop && A_lvl_4.idx[A_lvl_4_q] < phase_start_3
                        #    A_lvl_4_q += 1
                        #end
                        #while A_lvl_6_q < A_lvl_6_q_stop && A_lvl_6.idx[A_lvl_6_q] < phase_start_3
                        #    A_lvl_6_q += 1
                        #end
                        #while k <= phase_stop_3
                        while A_lvl_4_q < A_lvl_4_q_stop && A_lvl_6_q < A_lvl_6_q_stop
                            #k_start_2 = k
                            A_lvl_4_i = A_lvl_4.idx[A_lvl_4_q]
                            A_lvl_6_i = A_lvl_6.idx[A_lvl_6_q]
                            #phase_start_4 = max(k_start_2)
                            phase_stop_4 = min(A_lvl_4_i, A_lvl_6_i)#, phase_stop_3)
                            #if phase_stop_4 >= phase_start_4
                                #k_2 = k
                                if A_lvl_4_i == phase_stop_4 && A_lvl_6_i == phase_stop_4
                                    c_val = c_val + true
                                end
                                A_lvl_4_q += A_lvl_4_i == phase_stop_4
                                A_lvl_6_q += A_lvl_6_i == phase_stop_4
                                #k = phase_stop_4 + 1
                            #end
                        end
                        k = phase_stop_3 + 1
                    end

                    A_lvl_2_q += 1
                else
                end
                j = phase_stop_2 + 1
            end
            j = phase_stop + 1
        end
        j_start = j
        phase_start_8 = max(j_start)
        phase_stop_8 = min(j_stop)
        if phase_stop_8 >= phase_start_8
            j_4 = j
            j = phase_stop_8 + 1
        end
    end
    (c = (Scalar){0, Int64}(c_val),)
end

main()