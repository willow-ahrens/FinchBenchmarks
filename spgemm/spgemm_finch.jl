using Finch
using BenchmarkTools

for z0 = (0, 0.0, false)
    A = Tensor(Dense(SparseList(Element(z0))))
    B = Tensor(Dense(SparseList(Element(z0))))
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))

    AT = Tensor(Dense(SparseList(Element(z))))
    BT = Tensor(Dense(SparseList(Element(z))))

    eval(@finch_kernel function spgemm_finch_inner_kernel(C, AT, B)
        C .= 0
        for j=_, i=_, k=_
            C[i, j] += AT[k, i] * B[k, j]
        end
        return C
    end)

    w = Tensor(SparseByteMap(Element(z)))
    eval(@finch_kernel function spgemm_finch_gustavson_kernel(C, w, A, B)
        C .= 0
        for j=_
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
        return C
    end)

    w = Tensor(SparseHash{2}(Element(z)))
    eval(@finch_kernel function spgemm_finch_outer_kernel(C, w, A, BT)
        w .= 0
        for k=_, j=_, i=_
            w[i, j] += A[i, k] * BT[j, k]
        end
        C .= 0
        for j=_, i=_
            C[i, j] = w[i, j]
        end
        return C
    end)

    w = Tensor(Dense(SparseByteMap(Element(z))))
    eval(@finch_kernel function spgemm_finch_outer_kernel(C, w, A, BT)
        w .= 0
        for k=_, j=_, i=_
            w[i, j] += A[i, k] * BT[j, k]
        end
        C .= 0
        for j=_, i=_
            C[i, j] = w[i, j]
        end
        return C
    end)

    w = Tensor(Dense(Dense(Element(z))))
    eval(@finch_kernel function spgemm_finch_outer_kernel(C, w, A, BT)
        w .= 0
        for k=_, j=_, i=_
            w[i, j] += A[i, k] * BT[j, k]
        end
        C .= 0
        for j=_, i=_
            C[i, j] = w[i, j]
        end
        return C
    end)
end

function spgemm_finch_inner_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w2D = Tensor(SparseHash{2}(Element(z)))
    AT = Tensor(Dense(SparseList(Element(z))))
    AT = copyto!(AT, swizzle(A, 2, 1))
    time = @belapsed spgemm_finch_inner_kernel($C, $AT, $B)
    C = spgemm_finch_inner_kernel(C, AT, B).C
    return (time = time, C = C)
end

function spgemm_finch_gustavson_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseByteMap(Element(z)))
    time = @belapsed spgemm_finch_gustavson_kernel($C, $w, $A, $B)
    C = spgemm_finch_gustavson_kernel(C, w, A, B).C
    return (time = time, C = C)
end

function spgemm_finch_outer_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseHash{2}(Element(z)))
    BT = Tensor(Dense(SparseList(Element(z))))
    BT = copyto!(BT, swizzle(B, 2, 1))
    time = @belapsed spgemm_finch_outer_kernel($C, $w, $A, $BT)
    C = spgemm_finch_outer_kernel(C, w, A, BT).C
    return (time = time, C = C)
end

function spgemm_finch_outer_bytemap_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(Dense(SparseByteMap(Element(z))))
    BT = Tensor(Dense(SparseList(Element(z))))
    BT = copyto!(BT, swizzle(B, 2, 1))
    time = @belapsed spgemm_finch_outer_kernel($C, $w, $A, $BT)
    C = spgemm_finch_outer_kernel(C, w, A, BT).C
    return (time = time, C = C)
end

function spgemm_finch_outer_dense_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(Dense(Dense(Element(z))))
    BT = Tensor(Dense(SparseList(Element(z))))
    BT = copyto!(BT, swizzle(B, 2, 1))
    time = @belapsed spgemm_finch_outer_kernel($C, $w, $A, $BT)
    C = spgemm_finch_outer_kernel(C, w, A, BT).C
    return (time = time, C = C)
end

function spgemm_finch(f, A, B)
    _A = Tensor(A)
    _B = Tensor(B)
    C = Ref{Any}()
    (time, C[]) = f(_A, _B)
    return (;time = time, C = C[])
end

spgemm_finch_inner(A, B) = spgemm_finch(spgemm_finch_inner_measure, A, B)
spgemm_finch_gustavson(A, B) = spgemm_finch(spgemm_finch_gustavson_measure, A, B)
spgemm_finch_outer(A, B) = spgemm_finch(spgemm_finch_outer_measure, A, B)
spgemm_finch_outer_bytemap(A, B) = spgemm_finch(spgemm_finch_outer_bytemap_measure, A, B)
spgemm_finch_outer_dense(A, B) = spgemm_finch(spgemm_finch_outer_dense_measure, A, B)