using SuiteSparseGraphBLAS

function spmv_suite_sparse_kernel(y, A, x)
    mul!(y, A, x)
end

function spmv_suite_sparse(y, A, x)
    _y = GBVector(y)
    _A = GBMatrix(A)
    _x = GBVector(x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_suite_sparse_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end

