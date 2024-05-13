using Finch
using BenchmarkTools

function spmv_finch_band_kernel_helper_row_maj(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A_band::Tensor{DenseLevel{Int64, SparseBandLevel{Int64, Vector{Int64}, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}})
    @inbounds @fastmath begin
            y_lvl = y.lvl
            y_lvl_2 = y_lvl.lvl
            y_lvl_val = y_lvl.lvl.val
            A_band_lvl = A_band.lvl
            A_band_lvl_2 = A_band_lvl.lvl
            A_band_lvl_ptr = A_band_lvl_2.ptr
            A_band_lvl_idx = A_band_lvl_2.idx
            A_band_lvl_ofs = A_band_lvl_2.ofs
            A_band_lvl_2_val = A_band_lvl_2.lvl.val
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            x_lvl.shape == A_band_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_band_lvl_2.shape))"))
            result = nothing
            Finch.resize_if_smaller!(y_lvl_val, A_band_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_band_lvl.shape)
            for j_4 = 1:A_band_lvl.shape
                y_lvl_q = (1 - 1) * A_band_lvl.shape + j_4
                A_band_lvl_q = (1 - 1) * A_band_lvl.shape + j_4
                A_band_lvl_2_r = A_band_lvl_ptr[A_band_lvl_q]
                A_band_lvl_2_r_stop = A_band_lvl_ptr[A_band_lvl_q + 1] - 1
                if A_band_lvl_2_r <= A_band_lvl_2_r_stop
                    A_band_lvl_2_i1 = A_band_lvl_idx[A_band_lvl_2_r]
                    A_band_lvl_2_q_stop = A_band_lvl_ofs[A_band_lvl_2_r + 1]
                    A_band_lvl_2_i_2 = A_band_lvl_2_i1 - ((A_band_lvl_2_q_stop - A_band_lvl_ofs[A_band_lvl_2_r]) - 1)
                    A_band_lvl_2_q_ofs = (A_band_lvl_2_q_stop - A_band_lvl_2_i1) - 1
                else
                    A_band_lvl_2_i_2 = 1
                    A_band_lvl_2_i1 = 0
                end
                phase_start_2 = max(1, A_band_lvl_2_i_2)
                phase_stop_2 = min(x_lvl.shape, A_band_lvl_2_i1)
                if phase_stop_2 >= phase_start_2
                    for i_6 = phase_start_2:phase_stop_2
                        x_lvl_q = (1 - 1) * x_lvl.shape + i_6
                        A_band_lvl_2_q = A_band_lvl_2_q_ofs + i_6
                        x_lvl_2_val = x_lvl_val[x_lvl_q]
                        A_band_lvl_3_val = A_band_lvl_2_val[A_band_lvl_2_q]
                        y_lvl_val[y_lvl_q] += A_band_lvl_3_val * x_lvl_2_val
                    end
                end
            end
            resize!(y_lvl_val, A_band_lvl.shape)
            result = (y = Tensor((DenseLevel){Int64}(y_lvl_2, A_band_lvl.shape)),)
        end
end

function spmv_finch_band_kernel_row_maj(y, A, x)
    spmv_finch_band_kernel_helper_row_maj(y, A, x)
    y
end

function spmv_finch_band_unsym_row_maj(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseBand(Element(0.0))))
    @finch mode=:fast begin
        _A .= 0
        for j=_, i=_
            _A[i, j] = A[j, i]
        end
    end

    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_band_kernel_row_maj($_y, $_A, $_x)
    return (;time = time, y = y[])
end

