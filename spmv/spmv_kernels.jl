y = Tensor(Dense(Element(0.0)))
A = Tensor(Dense(SparseList(Element(0.0))))
A_int8 = Tensor(Dense(SparseList(Element(Int8(0)))))
A_pattern = Tensor(Dense(SparseList(Pattern())))
A_vbl = Tensor(Dense(SparseVBLLevel(Element(0.0))))
A_vbl_int8 = Tensor(Dense(SparseVBLLevel(Element(Int8(0)))))
A_vbl_pattern = Tensor(Dense(SparseVBLLevel(Pattern())))
A_band = Tensor(Dense(SparseBand(Element(0.0))))
A_point = Tensor(Dense(SparsePoint(Element(0.0))))
A_point_pattern = Tensor(Dense(SparsePoint(Pattern())))
x = Tensor(Dense(Element(0.0)))
diag = Tensor(Dense(Element(0.0)))
diag_int8 = Tensor(Dense(Element(Int8(0))))
diag_pattern = Tensor(Dense(Pattern()))
y_j = Scalar(0.0)

# 0.92x slowdown
println(@finch_kernel mode=fastfinch function ssymv_finch_kernel_helper(y, A, x, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A[i, j]
                    if i <= j
                        y[i] += A_ij * x_j
                    end
                    if i < j
                        y_j[] += A_ij * x[i]
                    end
                end
            end
        end
        y[j] += y_j[]
    end
end) 

# 0.90x slowdown
println(@finch_kernel mode=fastfinch function ssymv_finch_kernel_helper(y, A, x, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A[i, j]
                    let ij_leq = (i <= j), ij_geq = (i >= j)
                        if ij_leq
                            y[i] += A_ij * x_j
                        end
                        if ij_leq && !ij_geq
                            y_j[] += A_ij * x[i]
                        end
                    end
                end
            end
        end
        y[j] += y_j[]
    end
end) 

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
    return y
end)

println(@finch_kernel mode=fastfinch function ssymv_finch_int8_kernel_helper(y, A_int8, x, diag_int8, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A_int8[i, j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag_int8[j] * x_j
        end
    end
    return y
end)

println(@finch_kernel mode=fastfinch function ssymv_finch_pattern_kernel_helper(y, A_pattern, x, diag_pattern, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A_pattern[i, j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag_pattern[j] * x_j
        end
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_pattern_kernel_helper(y, A_pattern, x)
    y .= 0
    for j = _, i = _
        y[i] += A_pattern[i, j] * x[j]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_pattern_kernel_helper_row_maj(y, A_pattern, x)
    y .= 0
    for j = _, i = _
        y[j] += A_pattern[i, j] * x[i]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function ssymv_finch_parallel_kernel_helper(y, A, x, diag, y_j)
    y .= 0
    for j = parallel(_)
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A[walk(i), j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag[j] * x_j
        end
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_kernel_helper(y, A, x)
    y .= 0
    for j = _, i = _
        y[i] += A[i, j] * x[j]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_kernel_helper_row_maj(y, A, x)
    y .= 0
    for j = _, i = _
        y[j] += A[i, j] * x[i]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function ssymv_finch_vbl_kernel_helper(y, A_vbl, x, diag, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A_vbl[i, j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag[j] * x_j
        end
    end
    return y
end)

println(@finch_kernel mode=fastfinch function ssymv_finch_vbl_int8_kernel_helper(y, A_vbl_int8, x, diag_int8, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A_vbl_int8[i, j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag_int8[j] * x_j
        end
    end
    return y
end)

println(@finch_kernel mode=fastfinch function ssymv_finch_vbl_pattern_kernel_helper(y, A_vbl_pattern, x, diag_pattern, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A_vbl_pattern[i, j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag_pattern[j] * x_j
        end
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_vbl_kernel_helper(y, A_vbl, x)
    y .= 0
    for j = _, i = _
        y[i] += A_vbl[i, j] * x[j]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_vbl_kernel_helper_row_maj(y, A_vbl, x)
    y .= 0
    for j = _, i = _
        y[j] += A_vbl[i, j] * x[i]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function ssymv_finch_band_kernel_helper(y, A_band, x, diag, y_j)
    y .= 0
    for j = _
        let x_j = x[j]
            y_j .= 0
            for i = _
                let A_ij = A_band[i, j]
                    y[i] += x_j * A_ij
                    y_j[] += A_ij * x[i]
                end
            end
            y[j] += y_j[] + diag[j] * x_j
        end
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_band_kernel_helper(y, A_band, x)
    y .= 0
    for j = _, i = _
        y[i] += A_band[i, j] * x[j]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_band_kernel_helper_row_maj(y, A_band, x)
    y .= 0
    for j = _, i = _
        y[j] += A_band[i, j] * x[i]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_point_kernel_helper(y, A_point, x)
    y .= 0
    for j = _, i = _
        y[i] += A_point[i, j] * x[j]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_point_kernel_helper_row_maj(y, A_point, x)
    y .= 0
    for j = _, i = _
        y[j] += A_point[i, j] * x[i]
    end
    return y
end)

println(@finch_kernel mode=fastfinch function spmv_finch_point_pattern_kernel_helper(y, A_point_pattern, x)
    y .= 0
    for j = _, i = _
        y[i] += A_point_pattern[i, j] * x[j]
    end
    return y
end)