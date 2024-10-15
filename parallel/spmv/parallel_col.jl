using Finch
using BenchmarkTools


function parallel_col_mul(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                spmv(_y, _A, _x)
        end samples = 1 evals = 1
        return (; time=time, y=_y)
end

function spmv(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
        @inbounds @fastmath(begin
                println("A = $A")
                println("x = $x")
                y_lvl = y.lvl # DenseLevel
                # y_lvl_2 = y_lvl.lvl # ElementLevel
                y_lvl_val = y_lvl.lvl.val # Vector{Float64}

                A_lvl = A.lvl # DenseLevel
                A_lvl_2 = A_lvl.lvl # SparseListLevel
                A_lvl_ptr = A_lvl_2.ptr # Vector{Int64}
                A_lvl_idx = A_lvl_2.idx # Vector{Int64}
                # A_lvl_3 = A_lvl_2.lvl # ElementLevel
                A_lvl_2_val = A_lvl_2.lvl.val # Vector{Float64}

                x_lvl = x.lvl # DenseLevel
                # x_lvl_2 = x_lvl.lvl # ElementLevel
                x_lvl_val = x_lvl.lvl.val # Vector{Float66}

                x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
                Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
                Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)

                (_, n) = size(A)
                for j = 1:n
                        for q in A_lvl_ptr[j]:A_lvl_ptr[j+1]-1
                                i = A_lvl_idx[q]
                                println(y_lvl_val[i])
                                y_lvl_val[i] += A_lvl_2_val[q] * x_lvl_val[j]
                        end
                end
        end)
end

# function spmv(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
#         @inbounds @fastmath(begin
#                 y_lvl = y.lvl
#                 y_lvl_2 = y_lvl.lvl
#                 y_lvl_val = y_lvl.lvl.val
#                 A_lvl = A.lvl
#                 A_lvl_2 = A_lvl.lvl
#                 A_lvl_ptr = A_lvl_2.ptr
#                 A_lvl_idx = A_lvl_2.idx
#                 A_lvl_2_val = A_lvl_2.lvl.val
#                 x_lvl = x.lvl
#                 x_lvl_val = x_lvl.lvl.val
#                 A_lvl.shape == x_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(A_lvl.shape) != $(x_lvl.shape))"))
#                 Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
#                 Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
#                 val = y_lvl_val
#                 y_lvl_val = (Finch).moveto(y_lvl_val, CPU(Threads.nthreads()))
#                 x_lvl_val = (Finch).moveto(x_lvl_val, CPU(Threads.nthreads()))
#                 A_lvl_ptr = (Finch).moveto(A_lvl_ptr, CPU(Threads.nthreads()))
#                 A_lvl_idx = (Finch).moveto(A_lvl_idx, CPU(Threads.nthreads()))
#                 A_lvl_2_val = (Finch).moveto(A_lvl_2_val, CPU(Threads.nthreads()))
#                 Threads.@threads for i_4 = 1:Threads.nthreads()
#                         Finch.@barrier begin
#                                 @inbounds @fastmath(begin
#                                         phase_start_2 = max(1, 1 + fld(A_lvl.shape * (i_4 + -1), Threads.nthreads()))
#                                         phase_stop_2 = min(A_lvl.shape, fld(A_lvl.shape * i_4, Threads.nthreads()))
#                                         if phase_stop_2 >= phase_start_2
#                                                 for j_6 = phase_start_2:phase_stop_2
#                                                         A_lvl_q = (1 - 1) * A_lvl.shape + j_6
#                                                         x_lvl_q = (1 - 1) * x_lvl.shape + j_6
#                                                         x_lvl_2_val = x_lvl_val[x_lvl_q]
#                                                         A_lvl_2_q = A_lvl_ptr[A_lvl_q]
#                                                         A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q+1]
#                                                         if A_lvl_2_q < A_lvl_2_q_stop
#                                                                 A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop-1]
#                                                         else
#                                                                 A_lvl_2_i1 = 0
#                                                         end
#                                                         phase_stop_3 = min(A_lvl_2.shape, A_lvl_2_i1)
#                                                         if phase_stop_3 >= 1
#                                                                 if A_lvl_idx[A_lvl_2_q] < 1
#                                                                         A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
#                                                                 end
#                                                                 while true
#                                                                         A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
#                                                                         if A_lvl_2_i < phase_stop_3
#                                                                                 A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
#                                                                                 y_lvl_q = (1 - 1) * A_lvl_2.shape + A_lvl_2_i
#                                                                                 y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
#                                                                                 A_lvl_2_q += 1
#                                                                         else
#                                                                                 phase_stop_5 = min(phase_stop_3, A_lvl_2_i)
#                                                                                 if A_lvl_2_i == phase_stop_5
#                                                                                         A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
#                                                                                         y_lvl_q = (1 - 1) * A_lvl_2.shape + phase_stop_5
#                                                                                         y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
#                                                                                         A_lvl_2_q += 1
#                                                                                 end
#                                                                                 break
#                                                                         end
#                                                                 end
#                                                         end
#                                                 end
#                                         end
#                                         phase_start_6 = max(1, 1 + fld(A_lvl.shape * i_4, Threads.nthreads()))
#                                         phase_stop_7 = A_lvl.shape
#                                         if phase_stop_7 >= phase_start_6
#                                                 phase_stop_7 + 1
#                                         end
#                                 end)
#                                 nothing
#                         end
#                 end
#                 resize!(val, A_lvl_2.shape)
#                 (y=Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
#         end)
# end

# function spmv(y::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}, A::Tensor{DenseLevel{Int64,SparseListLevel{Int64,Vector{Int64},Vector{Int64},ElementLevel{0.0,Float64,Int64,Vector{Float64}}}}}, x::Tensor{DenseLevel{Int64,ElementLevel{0.0,Float64,Int64,Vector{Float64}}}})
#         @inbounds @fastmath(begin
#                 y_lvl = y.lvl
#                 y_lvl_2 = y_lvl.lvl
#                 y_lvl_val = y_lvl.lvl.val
#                 A_lvl = A.lvl
#                 A_lvl_2 = A_lvl.lvl
#                 A_lvl_ptr = A_lvl_2.ptr
#                 A_lvl_idx = A_lvl_2.idx
#                 A_lvl_2_val = A_lvl_2.lvl.val
#                 x_lvl = x.lvl
#                 x_lvl_val = x_lvl.lvl.val
#                 x_lvl.shape == A_lvl.shape || throw(DimensionMismatch("mismatched dimension limits ($(x_lvl.shape) != $(A_lvl.shape))"))
#                 Finch.resize_if_smaller!(y_lvl_val, A_lvl_2.shape)
#                 Finch.fill_range!(y_lvl_val, 0.0, 1, A_lvl_2.shape)
#                 for j_4 = 1:x_lvl.shape
#                         x_lvl_q = j_4
#                         A_lvl_q = j_4
#                         x_lvl_2_val = x_lvl_val[x_lvl_q]
#                         A_lvl_2_q = A_lvl_ptr[A_lvl_q]
#                         A_lvl_2_q_stop = A_lvl_ptr[A_lvl_q+1]
#                         if A_lvl_2_q < A_lvl_2_q_stop
#                                 A_lvl_2_i1 = A_lvl_idx[A_lvl_2_q_stop-1]
#                         else
#                                 A_lvl_2_i1 = 0
#                         end
#                         phase_stop = min(A_lvl_2.shape, A_lvl_2_i1)
#                         if phase_stop >= 1
#                                 if A_lvl_idx[A_lvl_2_q] < 1
#                                         A_lvl_2_q = Finch.scansearch(A_lvl_idx, 1, A_lvl_2_q, A_lvl_2_q_stop - 1)
#                                 end
#                                 while true
#                                         A_lvl_2_i = A_lvl_idx[A_lvl_2_q]
#                                         if A_lvl_2_i < phase_stop
#                                                 A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
#                                                 y_lvl_q = A_lvl_2_i
#                                                 y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
#                                                 A_lvl_2_q += 1
#                                         else
#                                                 phase_stop_3 = min(phase_stop, A_lvl_2_i)
#                                                 if A_lvl_2_i == phase_stop_3
#                                                         A_lvl_3_val = A_lvl_2_val[A_lvl_2_q]
#                                                         y_lvl_q = phase_stop_3
#                                                         y_lvl_val[y_lvl_q] = x_lvl_2_val * A_lvl_3_val + y_lvl_val[y_lvl_q]
#                                                         A_lvl_2_q += 1
#                                                 end
#                                                 break
#                                         end
#                                 end
#                         end
#                 end
#                 resize!(y_lvl_val, A_lvl_2.shape)
#                 (y=Tensor((DenseLevel){Int64}(y_lvl_2, A_lvl_2.shape)),)
#         end)
# end


