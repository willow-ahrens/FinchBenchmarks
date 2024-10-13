using Finch
using BenchmarkTools

function spmv_finch_vbl_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
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
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_r < A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(A_lvl_2.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_lvl_idx[A_lvl_2_r] < 1
                        A_lvl_2_r = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_r, A_lvl_2_r_stop - 1)
                    end
                    while true
                        i_start_2 = i
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_r]
                        A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                        A_lvl_2_i_2 = A_lvl_2_i - (A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r])
                        A_lvl_2_q_ofs = (A_lvl_2_q_stop - A_lvl_2_i) - 1
                        if A_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_lvl_2_i_2)
                            if A_lvl_2_i >= phase_start_3
                                for i_8 = phase_start_3:A_lvl_2_i
                                    y_lvl_q = (1 - 1) * A_lvl_2.shape + i_8
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_8
                                    A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
                                end
                            end
                            A_lvl_2_r += A_lvl_2_i == A_lvl_2_i
                            i = A_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_11 = phase_start_6:phase_stop_5
                                    y_lvl_q = (1 - 1) * A_lvl_2.shape + i_11
                                    A_lvl_2_q = A_lvl_2_q_ofs + i_11
                                    A_lvl_3_val_2 = A_lvl_2_val[A_lvl_2_q]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val_2 + y_lvl_val[y_lvl_q]
                                end
                            end
                            A_lvl_2_r += phase_stop_5 == A_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, A_lvl_2.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
        end
end

function spmv_finch_vbl_kernel(y, A, x)
    spmv_finch_vbl_kernel_helper(y, A, x)
    y
end

function spmv_finch_vbl_unsym(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_vbl_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end
