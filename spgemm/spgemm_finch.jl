using Finch
using BenchmarkTools

for z0 = (0, 0.0, false)
    A = Tensor(Dense(SparseList(Element(z0))))
    B = Tensor(Dense(SparseList(Element(z0))))
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    CD = Tensor(Dense(Dense(Element(z))))

    AT = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseByteMap(Element(z)))
    w2D = Tensor(SparseHash{2}(Element(z)))
    BT = Tensor(Dense(SparseList(Element(z))))

eval(@finch_kernel function spgemm_finch_inner_kernel(C, AT, B)
    C .= 0;
    for j=_, i=_, k=_
        C[i, j] += AT[k, i] * B[k, j]
    end
end)

eval(@finch_kernel function spgemm_finch_gustavson_kernel(C, w, A, B)
	 C .= 0;
	 for j=_
		 w .= 0
		 for k=_, i=_; w[i] += A[i, k] * B[k, j] end
		 for i=_; C[i, j] = w[i] end
	 end
end)

eval(@finch_kernel function spgemm_finch_outer_kernel(w2D, A, BT)
	w2D .= 0;
	for k=_, j=_, i=_
		w2D[i, j] += A[i, k] * BT[j, k]
	end
end)

end

function spgemm_finch_inner_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w2D = Tensor(SparseHash{2}(Element(z)))
    AT = Tensor(Dense(SparseList(Element(z))))
    @finch mode=fastfinch (C .= 0; for k=_, i=_; C[k, i] = A[i, k] end)
    @finch mode=fastfinch (C .= 0; for k=_, i=_; C[k, i] = 0 end)
    @finch mode=fastfinch (w2D .= 0; for k=_, i=_; w2D[k, i] = A[i, k] end)
    @finch mode=fastfinch (AT .= 0; for i=_, k=_; AT[k, i] = w2D[k, i] end)
    time = @belapsed spgemm_finch_inner_kernel($C, $AT, $B)
    return (time = time, C = C)
end

function spgemm_finch_gustavson_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w = Tensor(SparseByteMap(Element(z)))
    @finch mode=fastfinch (C .= 0; for k=_, i=_; C[k, i] = A[i, k] end)
    @finch mode=fastfinch (C .= 0; for k=_, i=_; C[k, i] = 0 end)
    spgemm_finch_gustavson_kernel(C, w, A, B)
    @finch mode=fastfinch (C .= 0; for k=_, i=_; C[k, i] = 0 end)
    time = @belapsed spgemm_finch_gustavson_kernel($C, $w, $A, $B)
    return (time = time, C = C)
end

function spgemm_finch_outer_measure(A, B)
    z = default(A) * default(B) + false
    C = Tensor(Dense(SparseList(Element(z))))
    w2D = Tensor(SparseHash{2}(Element(z)))
    BT = Tensor(Dense(SparseList(Element(z))))
    @finch mode=fastfinch (w2D .= 0; for j=_, k=_; w2D[j, k] = B[k, j] end)
    @finch (BT .= 0; for k=_, j=_; BT[j, k] = w2D[j, k] end)
    time = @belapsed spgemm_finch_outer_kernel($w2D, $A, $BT)
    @finch (C .= 0; for j=_, i=_; C[i, j] = w2D[i, j] end)
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
