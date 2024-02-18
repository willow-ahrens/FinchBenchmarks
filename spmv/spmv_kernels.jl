y = Tensor(Dense(Element(0.0)))
A = Tensor(Dense(SparseList(Element(0.0))))
x = Tensor(Dense(Element(0.0)))
diag = Tensor(Dense(Element(0.0)))
y_j = Scalar(0.0)

# 0.92x slowdown
# println(@finch_kernel mode=fastfinch function ssymv_finch_kernel_helper(y, A, x, y_j)
#     y .= 0
#     for j = _
#         let x_j = x[j]
#             y_j .= 0
#             for i = _
#                 let A_ij = A[i, j]
#                     if i <= j
#                         y[i] += A_ij * x_j
#                     end
#                     if i < j
#                         y_j[] += A_ij * x[i]
#                     end
#                 end
#             end
#         end
#         y[j] += y_j[]
#     end
# end) 

# 0.90x slowdown
# println(@finch_kernel mode=fastfinch function ssymv_finch_kernel_helper(y, A, x, y_j)
#     y .= 0
#     for j = _
#         let x_j = x[j]
#             y_j .= 0
#             for i = _
#                 let A_ij = A[i, j]
#                     let ij_leq = (i <= j), ij_geq = (i >= j)
#                         if ij_leq
#                             y[i] += A_ij * x_j
#                         end
#                         if ij_leq && !ij_geq
#                             y_j[] += A_ij * x[i]
#                         end
#                     end
#                 end
#             end
#         end
#         y[j] += y_j[]
#     end
# end) 

# 1.30x speedup
println(@finch_kernel mode=fastfinch function ssymv_finch_kernel_helper(y, A, x, diag, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A[i, j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag[j] * x_j
        end
    end
end)