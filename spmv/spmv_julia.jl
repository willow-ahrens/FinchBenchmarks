function spmv_kernel(y, A, x)
    y = A*x
    (y,)
end

function spmv_julia(y, A, x)
    _y = Vector(y)
    _A = SparseMatrixCSC(A) 
    _x = Vector(x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end