using Finch
using BenchmarkTools

function spmv_finch_blocked_helper(y, A, x)
    b = 4
    block_A = Tensor(Dense(SparseHash{1}(Dense(Dense(Element(0.0))))))
    @finch begin
        block_A .= 0
        for i = _
            for j = _
                block_A[mod1(i, b), mod1(j, b), fld1(i, b), fld1(j, b)] = A[i, j]
            end
        end
    end

    block_x = Tensor(Dense(Dense(Element(0.0))))
    @finch begin 
        for j = _
            block_x[mod1(j, b), fld1(j, b)] = x[j, b]
        end
    end

    @finch begin
        for I = _
            for J = _
                for i = _
                    for j = _
                         y[i, I] += block_A[i, j, I, J] * block_x[j, J]
                    end
                end
            end
        end
    end
    y
end

function spmv_finch_blocked(y, A, x) 
    _y = Tensor(Dense(Element(0.0)), y)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _x = Tensor(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_finch_blocked_helper($_y, $_A, $_x)
    return (;time = time, y = y[])
end