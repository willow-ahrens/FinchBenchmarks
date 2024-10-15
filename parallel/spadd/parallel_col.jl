using Finch
using BenchmarkTools


function parallel_col_add(C, A, B)
        _C = Tensor(Dense(Separate(SparseList(Element(0.0)))), C)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        _B = Tensor(Dense(SparseList(Element(0.0))), B)
        time = @belapsed begin
                (_C, _A, _B) = $(_C, _A, _B)
                @finch mode = :fast begin
                        _C .= 0
                        for j = parallel(_), i = _
                                _C[i, j] = _A[i, j] + _B[i, j]
                        end
                end
        end
        return (; time=time, C=_C)
end
