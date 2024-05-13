using Finch
using BenchmarkTools

function spmv_finch_point_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A_point::Tensor{DenseLevel{Int64, SparsePointLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_point_lvl = A_point.lvl
            A_point_lvl_2 = A_point_lvl.lvl
            A_point_lvl_ptr = A_point_lvl_2.ptr
            A_point_lvl_idx = A_point_lvl_2.idx
            A_point_lvl_2_val = A_point_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_point_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_point_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_point_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_point_lvl.shape)
            for j_4 = 1:A_point_lvl.shape
                y_lvl_q = (1 - 1) * A_point_lvl.shape + j_4
                A_point_lvl_q = (1 - 1) * A_point_lvl.shape + j_4
                A_point_lvl_2_q = A_point_lvl_ptr[A_point_lvl_q]
                A_point_lvl_2_q_stop = A_point_lvl_ptr[A_point_lvl_q + 1]
                if A_point_lvl_2_q < A_point_lvl_2_q_stop
                    A_point_lvl_2_i = A_point_lvl_idx[A_point_lvl_2_q]
                else
                    A_point_lvl_2_i = 0
                end
                phase_stop = min(x_lvl.shape, A_point_lvl_2_i)
                if phase_stop >= 1
                    if phase_stop < A_point_lvl_2_i
                    else
                        A_point_lvl_3_val = A_point_lvl_2_val[A_point_lvl_2_q]
                        x_lvl_q = (1 - 1) * x_lvl.shape + phase_stop
                        x_lvl_2_val = x_lvl_val[x_lvl_q]
                        y_lvl_val[y_lvl_q] += A_point_lvl_3_val * x_lvl_2_val
                    end
                end
            end
            resize!(y_lvl_val, A_point_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_point_lvl.shape)),)
        end
end

function spmv_finch_point_kernel_row_maj(y, A, x)
    spmv_finch_point_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_point_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparsePoint(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end
    
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_point_kernel_row_maj($_y, $_A, $_x)
    return (;time = time, y = y[])
end
