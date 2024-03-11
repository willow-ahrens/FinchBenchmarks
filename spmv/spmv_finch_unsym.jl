using Finch
using BenchmarkTools

function spmv_finch_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_2_val = A_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
            for j_4 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(A_lvl_2.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            y_lvl_q = (1 - 1) * A_lvl_2.shape + A_lvl_2_i
                            y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                y_lvl_q = (1 - 1) * A_lvl_2.shape + phase_stop_3
                                y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_kernel(y, A, x)
    spmv_finch_kernel_helper(y, A, x)
    y
end

function spmv_finch_unsym(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end