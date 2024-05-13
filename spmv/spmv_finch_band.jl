using Finch
using BenchmarkTools

function ssymv_finch_band_kernel_helper(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparseBandLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, diag::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, y_j::Scalar{0.0, Float64})
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
            diag_lvl = diag.lvl
            diag_lvl_val = diag_lvl.lvl.val
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
                diag_lvl_q = (1 - 1) * diag_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                diag_lvl_2_val = diag_lvl_val[diag_lvl_q]
                y_j_val = 0
                A_lvl_2_r = A_lvl_ptr[A_lvl_q]
                A_lvl_2_r_stop = A_lvl_ptr[A_lvl_q + 1] - 1
                if A_lvl_2_r <= A_lvl_2_r_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_r]
                    A_lvl_2_q_stop = A_lvl_ofs[A_lvl_2_r + 1]
                    A_lvl_2_i_2 = A_lvl_2_i1 - ((A_lvl_2_q_stop - A_lvl_ofs[A_lvl_2_r]) - 1)
                    A_lvl_2_q_ofs = (A_lvl_2_q_stop - A_lvl_2_i1) - 1
                else
                    A_lvl_2_i_2 = 1
                    A_lvl_2_i1 = 0
                end
                phase_start_2 = max(1, A_lvl_2_i_2)
                phase_stop_2 = min(x_lvl.shape, A_lvl_2_i1)
                if phase_stop_2 >= phase_start_2
                    for i_8 = phase_start_2:phase_stop_2
                        A_lvl_2_q = A_lvl_2_q_ofs + i_8
                        y_lvl_q = (1 - 1) * A_lvl.shape + i_8
                        x_lvl_q_2 = (1 - 1) * x_lvl.shape + i_8
                        A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                        x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                        y_lvl_val[y_lvl_q] = A_lvl_3_val * x_lvl_2_val + y_lvl_val[y_lvl_q]
                        y_j_val = A_lvl_3_val * x_lvl_2_val_2 + y_j_val
                    end
                end
                y_j.val = y_j_val
                y_lvl_val[y_lvl_q_2] = y_j_val + y_lvl_val[y_lvl_q_2] + x_lvl_2_val * diag_lvl_2_val
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_band_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    ssymv_finch_band_kernel_helper(y, A, x, d, y_j)
    y
end

function spmv_finch_band(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))), A)
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
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_band_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end
