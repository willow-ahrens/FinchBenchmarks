using Finch
using BenchmarkTools

function spmv_finch_point_pattern_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Tensor{DenseLevel{Int64, SparsePointLevel{Int64, Vector{Int64}, Vector{Int64}, PatternLevel{Int64}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_4 = 1:A_lvl.shape
                y_lvl_q = (1 - 1) * A_lvl.shape + j_4
                A_lvl_q = (1 - 1) * A_lvl.shape + j_4
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                else
                    A_lvl_2_i = 0
                end
                phase_stop = min(x_lvl.shape, A_lvl_2_i)
                if phase_stop >= 1
                    if phase_stop < A_lvl_2_i
                    else
                        x_lvl_q = (1 - 1) * x_lvl.shape + phase_stop
                        x_lvl_2_val = x_lvl_val[x_lvl_q]
                        y_lvl_val[y_lvl_q] = x_lvl_2_val + y_lvl_val[y_lvl_q]
                    end
                end
            end
            resize!(y_lvl_val, A_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function spmv_finch_point_pattern_kernel_row_maj(y, A, x)
    spmv_finch_point_pattern_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_point_pattern_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Element(0.0))), A)
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    
    A_pattern = pattern!(_A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_point_pattern_kernel_row_maj($_y, $A_pattern, $_x)
    return (;time = time, y = y[])
end
