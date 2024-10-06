using Finch
using BenchmarkTools


function reference_mul(y, A, x)
        _y = Tensor(Dense(Element(0.0)), y)
        _A = Tensor(Dense(SparseList(Element(0.0))), A)
        _x = Tensor(Dense(Element(0.0)), x)
        time = @belapsed begin
                (_y, _A, _x) = $(_y, _A, _x)
                @finch mode = :fast begin
                        _y .= 0
                        for j = _, i = _
                                _y[i] = _A[i, j] * _x[j]
                        end
                end
        end
        return (; time=time, y=_y)
end
