using Finch
using BenchmarkTools

function ssymv_finch_kernel_helper(y::Fiber{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, A::Fiber{DenseLevel{Int64, SparseListLevel{Int64, Vector{Int64}, Vector{Int64}, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}}, x::Fiber{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, d::Fiber{DenseLevel{Int64, ElementLevel{0.0, Float64, Int64, Vector{Float64}}}}, y_j::Scalar{0.0, Float64})
    @inbounds begin
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
            d_lvl = d.lvl
            d_lvl_val = d_lvl.lvl.val
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            x_lvl.shape == A_lvl_2.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl_2.shape))"))
            A_lvl.shape == d_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(d_lvl.shape))"))
            A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
            Finch.resize_if_smaller!(y_lvl_val, A_lvl.shape)
            Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl.shape)
            for j_6 = 1:A_lvl.shape
                x_lvl_q = (1 - 1) * x_lvl.shape + j_6
                A_lvl_q = (1 - 1) * A_lvl.shape + j_6
                y_lvl_q_2 = (1 - 1) * A_lvl.shape + j_6
                d_lvl_q = (1 - 1) * d_lvl.shape + j_6
                x_lvl_2_val = x_lvl_val[x_lvl_q]
                d_lvl_2_val = d_lvl_val[d_lvl_q]
                y_j_val = 0
                A_lvl_2_q = A_lvl_ptr[A_lvl_q]
                A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q + 1]
                if A_lvl_2_q < A_lvl_2_q_stop
                    A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop - 1]
                else
                    A_lvl_2_i1 = 0
                end
                phase_stop = (min)(x_lvl.shape, A_lvl_2_i1)
                if phase_stop >= 1
                    i = 1
                    if A_lvl_idx[A_lvl_2_q] < 1
                        A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
                    end
                    while i <= phase_stop
                        A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
                        phase_stop_2 = A_lvl_2_i
                        # phase_stop_2 = (min)(phase_stop, A_lvl_2_i)
                        # if A_lvl_2_i == phase_stop_2
                            A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
                            y_lvl_q = (1 - 1) * A_lvl.shape + phase_stop_2
                            x_lvl_q_2 = (1 - 1) * x_lvl.shape + phase_stop_2
                            x_lvl_2_val_2 = x_lvl_val[x_lvl_q_2]
                            y_lvl_val[y_lvl_q] = (+)((*)(A_lvl_3_val, x_lvl_2_val), y_lvl_val[y_lvl_q])
                            y_j_val = (+)((*)(A_lvl_3_val, x_lvl_2_val_2), y_j_val)
                            A_lvl_2_q += 1
                        # end
                        i = phase_stop_2 + 1
                    end
                end
                y_lvl_val[y_lvl_q_2] = (+)(d_lvl_2_val, y_lvl_val[y_lvl_q_2], y_j_val)
            end
            qos = 1 * A_lvl.shape
            resize!(y_lvl_val, qos)
            (y = Fiber((DenseLevel){Int64}(y_lvl_2, A_lvl.shape)),)
        end
end

function ssymv_finch_kernel(y, A, x, d)
    y_j = Scalar(0.0)
    
    # println(@finch_kernel mode=fastfinch function ssymv_finch_kernel_helper(y, A, x, d, y_j)
    #     y .= 0
    #     for j = _
    #         let x_j = x[j]
    #             y_j .= 0
    #             for i = _
    #                 let A_ij = A[i, j]
    #                     y[i] += x_j * A_ij
    #                     y_j[] += A_ij * x[i]
    #                 end
    #             end
    #         end
    #         y[j] += y_j[] + d[j]
    #     end
    # end)
    
    ssymv_finch_kernel_helper(y, A, x, d, y_j)

    #=
    @finch mode=fastfinch begin
        y .= 0
        for j = _
            let x_j = x[j]
                y_j .= 0
                for i = _
                    let A_ij = A[i, j]
                        if i <= j
                            y[i] += x_j * A_ij
                        end
                        if i < j
                            y_j[] += A_ij * x[i]
                        end
                    end
                end
            end
            y[j] += y_j[]
        end
    end
    =#
    y
end

using Cthulhu

function spmv_finch(y, A, x) 
    _y = Fiber!(Dense(Element(0.0)), y)
    _A = Fiber!(Dense(SparseList(Element(0.0))))
    _d = Fiber!(Dense(Element(0.0)))
    @finch (_A .= 0; for j = _, i = _; if i < j; _A[i, j] = A[i, j] end end)
    @finch (_d .= 0; for j = _, i = _; if i == j; _d[i] = A[i, j] * x[i] end end)
    @info "pruning" nnz(A) nnz(_A)
    
    _x = Fiber!(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_kernel($_y, $_A, $_x, $_d)
    return (;time = time, y = y[])
end