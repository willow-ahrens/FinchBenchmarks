using Finch
using BenchmarkTools

function assign_block_A(A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, block_A::Tensor{DenseLevel{Int64, DenseLevel{Int64, DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}}}, b::Int64)
    @inbounds @fastmath begin
            A_lvl = A.lvl
            A_lvl_2 = A_lvl.lvl
            A_lvl_ptr = A_lvl_2.ptr
            A_lvl_idx = A_lvl_2.idx
            A_lvl_2_val = A_lvl_2.lvl.val
            block_A_lvl = block_A.lvl
            block_A_lvl_2 = block_A_lvl.lvl
            block_A_lvl_3 = block_A_lvl_2.lvl
            block_A_lvl_4 = block_A_lvl_3.lvl
            block_A_lvl_4_val = block_A_lvl_4.lvl.val
            pos_stop = block_A_lvl_4.shape * block_A_lvl_3.shape * block_A_lvl_2.shape * block_A_lvl.shape
            Finch.resize_if_smaller!(block_A_lvl_4_val, pos_stop)
            Finch.fill_range!(block_A_lvl_4_val, 0.0, 1, pos_stop)
            for j_5 = 1:A_lvl.shape
                A_lvl_q = (1 - 1) * A_lvl.shape + j_5
                v_5 = fld1(j_5, b)
                block_A_lvl_q = (1 - 1) * block_A_lvl.shape + v_5
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = min(A_lvl_2_i1, A_lvl_2.shape)
                if phase_stop >= 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while true
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        if A_lvl_2_i < phase_stop
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            v_9 = fld1(A_lvl_2_i, b)
                            block_A_lvl_2_q = (block_A_lvl_q - 1) * block_A_lvl_2.shape + v_9
                            v_10 = mod1(j_5, b)
                            block_A_lvl_3_q = (block_A_lvl_2_q - 1) * block_A_lvl_3.shape + v_10
                            v_11 = mod1(A_lvl_2_i, b)
                            block_A_lvl_4_q = (block_A_lvl_3_q - 1) * block_A_lvl_4.shape + v_11
                            block_A_lvl_4_val[block_A_lvl_4_q] = A_lvl_3_val
                            A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(A_lvl_2_i, phase_stop)
                            if A_lvl_2_i == phase_stop_3
                                A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                                v_15 = fld1(phase_stop_3, b)
                                block_A_lvl_2_q = (block_A_lvl_q - 1) * block_A_lvl_2.shape + v_15
                                v_16 = mod1(j_5, b)
                                block_A_lvl_3_q_2 = (block_A_lvl_2_q - 1) * block_A_lvl_3.shape + v_16
                                v_17 = mod1(phase_stop_3, b)
                                block_A_lvl_4_q_2 = (block_A_lvl_3_q_2 - 1) * block_A_lvl_4.shape + v_17
                                block_A_lvl_4_val[block_A_lvl_4_q_2] = A_lvl_3_val
                                A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
            end
            resize!(block_A_lvl_4_val, block_A_lvl_4.shape * block_A_lvl_3.shape * block_A_lvl_2.shape * block_A_lvl.shape)
            nothing
        end
end

function assign_block_x(x::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, block_x::Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, b::Int64)
    @inbounds @fastmath begin
            x_lvl = x.lvl
            x_lvl_val = x_lvl.lvl.val
            block_x_lvl = block_x.lvl
            block_x_lvl_2 = block_x_lvl.lvl
            block_x_lvl_2_val = block_x_lvl_2.lvl.val
            pos_stop = block_x_lvl_2.shape * block_x_lvl.shape
            Finch.resize_if_smaller!(block_x_lvl_2_val, pos_stop)
            Finch.fill_range!(block_x_lvl_2_val, 0.0, 1, pos_stop)
            for j_5 = 1:x_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_5
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                v_3 = fld1(j_5, b)
                block_x_lvl_q = (1 - 1) * block_x_lvl.shape + v_3
                v_4 = mod1(j_5, b)
                block_x_lvl_2_q = (block_x_lvl_q - 1) * block_x_lvl_2.shape + v_4
                block_x_lvl_2_val[block_x_lvl_2_q] = x_lvl_2_val
            end
            resize!(block_x_lvl_2_val, block_x_lvl_2.shape * block_x_lvl.shape)
            nothing
        end
end

function blocked_spmv_kernel(y::Tensor{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, block_A::Tensor{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}}}, block_x::Tensor{DenseLevel{Int64, DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, b::Int64)
    @inbounds begin
            y_lvl = y.lvl
            y_lvl_val = y_lvl.lvl.val
            block_A_lvl = block_A.lvl
            block_A_lvl_2 = block_A_lvl.lvl
            block_A_lvl_ptr = block_A_lvl_2.ptr
            block_A_lvl_idx = block_A_lvl_2.idx
            block_A_lvl_3 = block_A_lvl_2.lvl
            block_A_lvl_4 = block_A_lvl_3.lvl
            block_A_lvl_4_val = block_A_lvl_4.lvl.val
            block_x_lvl = block_x.lvl
            block_x_lvl_2 = block_x_lvl.lvl
            block_x_lvl_2_val = block_x_lvl_2.lvl.val
            block_x_lvl_2.shape == block_A_lvl_3.shape || throw(DimensionMismatch("mismatched dimension limits ($(block_x_lvl_2.shape) != $(block_A_lvl_3.shape))"))
            block_x_lvl.shape == block_A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(block_x_lvl.shape) != $(block_A_lvl.shape))"))
            Finch.resize_if_smaller!(y_lvl_val, y_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, y_lvl.shape)
            for J_4 = 1:block_x_lvl.shape
                block_x_lvl_q = (1 - 1) * block_x_lvl.shape + J_4
                block_A_lvl_q = (1 - 1) * block_A_lvl.shape + J_4
                block_A_lvl_2_q = block_A_lvl_ptr[block_A_lvl_q]
                block_A_lvl_2_q_stop = block_A_lvl_ptr[block_A_lvl_q + 1]
                if block_A_lvl_2_q < block_A_lvl_2_q_stop
                    block_A_lvl_2_i1 = block_A_lvl_idx[block_A_lvl_2_q_stop - 1]
                else
                    block_A_lvl_2_i1 = 0
                end
                phase_stop = min(block_A_lvl_2_i1, block_A_lvl_2.shape)
                if phase_stop >= 1
                    if block_A_lvl_idx[block_A_lvl_2_q] < 1
                        block_A_lvl_2_q = Finch.scansearch(block_A_lvl_idx, 1, block_A_lvl_2_q, block_A_lvl_2_q_stop - 1)
                    end
                    while true
                        block_A_lvl_2_i = block_A_lvl_idx[block_A_lvl_2_q]
                        if block_A_lvl_2_i < phase_stop
                            for j_4 = 1:block_x_lvl_2.shape
                                block_x_lvl_2_q = (block_x_lvl_q - 1) * block_x_lvl_2.shape + j_4
                                block_A_lvl_3_q = (block_A_lvl_2_q - 1) * block_A_lvl_3.shape + j_4
                                block_x_lvl_3_val = block_x_lvl_2_val[block_x_lvl_2_q]
                                for i_5 = 1:block_A_lvl_4.shape
                                    block_A_lvl_4_q = (block_A_lvl_3_q - 1) * block_A_lvl_4.shape + i_5
                                    block_A_lvl_5_val = block_A_lvl_4_val[block_A_lvl_4_q]
                                    _i_2 = i_5 + b * (block_A_lvl_2_i + -1)
                                    y_lvl_q = (1 - 1) * y_lvl.shape + _i_2
                                    y_lvl_val[y_lvl_q] += block_A_lvl_5_val * block_x_lvl_3_val
                                end
                            end
                            block_A_lvl_2_q += 1
                        else
                            phase_stop_3 = min(block_A_lvl_2_i, phase_stop)
                            if block_A_lvl_2_i == phase_stop_3
                                for j_5 = 1:block_x_lvl_2.shape
                                    block_x_lvl_2_q = (block_x_lvl_q - 1) * block_x_lvl_2.shape + j_5
                                    block_A_lvl_3_q = (block_A_lvl_2_q - 1) * block_A_lvl_3.shape + j_5
                                    block_x_lvl_3_val_2 = block_x_lvl_2_val[block_x_lvl_2_q]
                                    for i_7 = 1:block_A_lvl_4.shape
                                        block_A_lvl_4_q_2 = (block_A_lvl_3_q - 1) * block_A_lvl_4.shape + i_7
                                        block_A_lvl_5_val_2 = block_A_lvl_4_val[block_A_lvl_4_q_2]
                                        _i_4 = i_7 + b * (phase_stop_3 + -1)
                                        y_lvl_q = (1 - 1) * y_lvl.shape + _i_4
                                        y_lvl_val[y_lvl_q] += block_A_lvl_5_val_2 * block_x_lvl_3_val_2
                                    end
                                end
                                block_A_lvl_2_q += 1
                            end
                            break
                        end
                    end
                end
            end
            resize!(y_lvl_val, y_lvl.shape)
            nothing
        end
end

function spmv_finch_blocked_helper(y, block_A, block_x, b)
    # @finch mode=fastfinch begin
    #     y .= 0
    #     for J = _
    #         for I = _
    #             for j = _
    #                 for i = _
    #                     let _i = (I - 1) * b + i
    #                         y[_i] += block_A[i, j, I, J] * block_x[j, J]
    #                     end
    #                 end
    #             end
    #         end
    #     end
    # end
    blocked_spmv_kernel(y, block_A, block_x, b)
    y
end

function spmv_finch_blocked(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    y = Ref{Any}()

    b = 8
    (n, m) = size(A)
    _block_A = Tensor(Dense(SparseHash{1}(Dense(Dense(Element(0.0))))), b, b, fld1(n, b), fld1(m, b))
    @finch mode=fastfinch begin
        _block_A .= 0
        for j = _
            for i = _
                _block_A[mod1(i, b), mod1(j, b), fld1(i, b), fld1(j, b)] = A[i, j]
            end
        end
    end
    block_A = Tensor(Dense(SparseList(Dense(Dense(Element(0.0))))), _block_A)
    # assign_block_A(A, block_A, b)
    @info "blocked inflation" countstored(block_A) / countstored(_A)

    block_x = Tensor(Dense(Dense(Element(0.0))), b, fld1(n, b))
    @finch mode=fastfinch begin 
        block_x .= 0
        for j = _
            block_x[mod1(j, b), fld1(j, b)] = x[j]
        end
    end
    # assign_block_x(x, block_x, b)

    time = @belapsed $y[] = spmv_finch_blocked_helper($_y, $block_A, $block_x, $b)
    return (;time = time, y = y[])
end