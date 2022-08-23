using Finch
using SparseArrays
using BenchmarkTools

using MatrixDepot
include("TensorMarket.jl")
using .TensorMarket


function spmv_taco(A, x)
    y_ref = fiber(x)
    @finch @loop i j y_ref[i] += A[i, j] * x[j]
    @finch @loop i y_ref[i] = 0

    @info :y
    ttwrite("y.ttx", ffindnz(y_ref)..., size(y_ref))
    (I, J, V) = findnz(A)
    @info :A
    ttwrite("A.ttx", (I, J), V, size(A))
    @info :x
    ttwrite("x.ttx", ffindnz(x)..., size(x))

    io = IOBuffer()

    @info :run
    run(pipeline(`./spmv_taco y.ttx A.ttx x.ttx`, stdout=io))

    y = fsparse(ttread("y.ttx")...)

    @finch @loop i j y_ref[i] += A[i, j] * x[j]

    #@assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

function spmv_finch(_A, x)
    A = fiber(_A)
    y = fiber(x)
    x = fiber(x)
    println(@finch_code @loop i j y[i] += A[i, j] * x[j])
    return @belapsed (A = $A; x = $x; y = $y; @finch @loop i j y[i] += A[i, j] * x[j])
end

foo(y, A, x) = @inbounds begin
    y_lvl = y.lvl
    y_lvl_2 = y_lvl.lvl
    y_lvl_2_val_alloc = length(y_lvl.lvl.val)
    y_lvl_2_val = 0.0
    A_lvl = A.lvl
    A_lvl_2 = A_lvl.lvl
    A_lvl_2_pos_alloc = length(A_lvl_2.pos)
    A_lvl_2_idx_alloc = length(A_lvl_2.idx)
    A_lvl_3 = A_lvl_2.lvl
    A_lvl_3_val_alloc = length(A_lvl_2.lvl.val)
    A_lvl_3_val = 0.0
    x_lvl = x.lvl
    x_lvl_2 = x_lvl.lvl
    x_lvl_2_val_alloc = length(x_lvl.lvl.val)
    x_lvl_2_val = 0.0
    j_stop = A_lvl_2.I
    i_stop = A_lvl.I
    y_lvl_2_val_alloc = (Finch).refill!(y_lvl_2.val, 0.0, 0, 4)
    y_lvl_2_val_alloc < 1 * A_lvl.I && (y_lvl_2_val_alloc = (Finch).refill!(y_lvl_2.val, 0.0, y_lvl_2_val_alloc, 1 * A_lvl.I))
    for i = 1:i_stop
        y_lvl_q = (1 - 1) * A_lvl.I + i
        A_lvl_q = (1 - 1) * A_lvl.I + i
        y_lvl_2_val = y_lvl_2.val[y_lvl_q]
        A_lvl_2_q = A_lvl_2.pos[A_lvl_q]
        A_lvl_2_q_stop = A_lvl_2.pos[A_lvl_q + 1]
        if A_lvl_2_q < A_lvl_2_q_stop
            A_lvl_2_i = A_lvl_2.idx[A_lvl_2_q]
            A_lvl_2_i1 = A_lvl_2.idx[A_lvl_2_q_stop - 1]
        else
            A_lvl_2_i = 1
            A_lvl_2_i1 = 0
        end
        j = 1
        j_start = j
        phase_start = max(j_start)
        phase_stop = min(A_lvl_2_i1, j_stop)
        if phase_stop >= phase_start
            j = j
            j = phase_start
            while A_lvl_2_q < A_lvl_2_q_stop && A_lvl_2.idx[A_lvl_2_q] < phase_start
                A_lvl_2_q += 1
            end
            while j <= phase_stop
                j_start_2 = j
                A_lvl_2_i = A_lvl_2.idx[A_lvl_2_q]
                phase_stop_2 = min(A_lvl_2_i, phase_stop)
                j_2 = j
                if A_lvl_2_i == phase_stop_2
                    A_lvl_3_val = A_lvl_3.val[A_lvl_2_q]
                    j_3 = phase_stop_2
                    x_lvl_q = (1 - 1) * x_lvl.I + j_3
                    x_lvl_2_val = x_lvl_2.val[x_lvl_q]
                    y_lvl_2_val = y_lvl_2_val + A_lvl_3_val * x_lvl_2_val
                    A_lvl_2_q += 1
                else
                end
                j = phase_stop_2 + 1
            end
            j = phase_stop + 1
        end
        j_start = j
        phase_start_3 = max(j_start)
        phase_stop_3 = min(j_stop)
        if phase_stop_3 >= phase_start_3
            j_4 = j
            j = phase_stop_3 + 1
        end
        y_lvl_2.val[y_lvl_q] = y_lvl_2_val
    end
    (y = Fiber((Finch.DenseLevel){Int64}(A_lvl.I, y_lvl_2), (Finch.Environment)(; name = :y)),)
end

function spmv_foo(_A, x)
    A = fiber(_A)
    y = fiber(x)
    x = fiber(copy(x))
    return @belapsed foo($y, $A, $x)
end


@info "loading"
A = SparseMatrixCSC(matrixdepot("Boeing/ct20stif"))
(m, n) = size(A)
#@info "taco"
#println("taco_time: ", spmv_taco(A, rand(n)))
#@info "finch"
#println("finch_time: ", spmv_finch(A, rand(n)))
println("finch_time: ", spmv_foo(A, rand(n)))
