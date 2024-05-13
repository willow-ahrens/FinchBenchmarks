using Finch
using BenchmarkTools

function ssymv_finch_vbl_pattern_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseVBLLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, PatternLevel{Int64}}}, y_j::Scalar{0.0, Float64})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_ofs = A_lvl_2.ofs
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            diag_lvl = diag.lvl
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == diag_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(diag_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                y_j_val = 0
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_r < A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i1)
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
                        if A_lvl_2_i < phase_stop
                            phase_start_3 = max(i_start_2, 1 + A_lvl_2_i_2)
                            if A_lvl_2_i >= phase_start_3
                                for i_10 = phase_start_3:A_lvl_2_i
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_10
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_10
                                    x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val = x_lvl_2_val_2 + y_j_val
                                end
                            end
                            A_lvl_2_r += A_lvl_2_i == A_lvl_2_i
                            i = A_lvl_2_i + 1
                        else
                            phase_start_4 = i
                            phase_stop_5 = min(A_lvl_2_i, phase_stop)
                            phase_start_6 = max(1 + A_lvl_2_i_2, phase_start_4)
                            if phase_stop_5 >= phase_start_6
                                for i_14 = phase_start_6:phase_stop_5
                                    y_lvl_q = (1 - 1) * A_lvl.shape + i_14
                                    x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_14
                                    x_lvl_2_val_3 = x_lvl_val[x_lvl_q_2]
                                    y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                                    y_j_val += x_lvl_2_val_3
                                end
                            end
                            A_lvl_2_r += phase_stop_5 == A_lvl_2_i
                            i = phase_stop_5 + 1
                            break
                        end
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = x_lvl_2_val + y_lvl_val[y_lvl_q_2] + y_j_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_vbl_pattern_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_vbl_pattern_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_vbl_pattern(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseVBLLevel(Element(0.0))), A)
    _d = Tensor(Dense(Element(0.0)))
    @finch mode=:fast begin
        _A .= 0
        _d .= 0
        for j = _, i = _
            if i < j
                _A[i, j] = A[i, j]
            end
            if i == j
                _d[i] = A[i, j]
            end
        end
    end
    # @info "pruning" nnz(A) nnz(_A)
    @info "memory footprint" Base.summarysize(_A)
    
    A_pattern = pattern!(_A)
    d_pattern = pattern!(_d)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_vbl_pattern_kernel($_y, $A_pattern, $_x, $d_pattern)
    return (;time = time, y = y[])
end
