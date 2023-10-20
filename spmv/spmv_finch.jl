using Finch
using BenchmarkTools

function ssymv_finch_kernel(y, A, x)
    y_j = Scalar(0.0)
    @finch begin
        y .= 0
        for j = _
            let x_j = x[j]
                y_j .= 0
                for i = _
                    let A_ij = A[i, j]
                        if uptrimask[i, j]
                            y[i] += x_j * A_ij
                        end
                        if uptrimask[i, j - 1]
                            y_j[] += A_ij * x[i]
                        end
                    end
                end
            end
            y[j] += y_j[]
        end
    end
    y
end

function spmv_finch(y, A, x) 
    _y = Fiber!(Dense(Element(0.0)), y)
    _A = Fiber!(Dense(SparseList(Element(0.0))), A)
    _x = Fiber!(Dense(Element(0.0)), x)
    y = Ref{Any}()
    time = @belapsed $y[] = ssymv_finch_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end