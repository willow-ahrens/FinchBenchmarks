using Finch
using BenchmarkTools

function spgemm_finch_inner_kernel(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseHash{2}(Element(z)))
    AT = Tensor(Dense(SparseList(Element(z))))
    @finch mode=fastfinch (w .= 0; for k=_, i=_; w[k, i] = A[i, k] end)
    @finch mode=fastfinch (AT .= 0; for i=_, k=_; AT[k, i] = w[k, i] end)
    @finch (C .= 0; for j=_, i=_, k=_; C[i, j] += AT[k, i] * B[k, j] end)
    return C
end

function spgemm_finch_gustavson_kernel(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseByteMap(Element(z)))
    @finch begin
        C .= 0
        for j=_
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
    end
    return C
end

function spgemm_finch_outer_kernel(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseHash{2}(Element(z)))
    BT = Tensor(Dense(SparseList(Element(z))))
    @finch mode=fastfinch (w .= 0; for j=_, k=_; w[j, k] = B[k, j] end)
    @finch (BT .= 0; for k=_, j=_; BT[j, k] = w[j, k] end)
    @finch (w .= 0; for k=_, j=_, i=_; w[i, j] += A[i, k] * BT[j, k] end)
    @finch (C .= 0; for j=_, i=_; C[i, j] = w[i, j] end)
    return C
end

function spgemm_finch(f, A, B)
    _A = Tensor(A)
    _B = Tensor(B)
    C = Ref{Any}()
    time = @belapsed $C[] = $f($_A, $_B)
    return (;time = time, C = C[])
end

spgemm_finch_inner(A, B) = spgemm_finch(spgemm_finch_inner_kernel, A, B)
spgemm_finch_gustavson(A, B) = spgemm_finch(spgemm_finch_gustavson_kernel, A, B)
spgemm_finch_outer(A, B) = spgemm_finch(spgemm_finch_outer_kernel, A, B)