using SuiteSparseGraphBLAS

function spmv_suite_sparse_kernel(y, A, x)
    y .= 0
    mul!(y, A, x; accum=+)
end

function spmv_suite_sparse(y, A, x)
    gbset(:nthreads, 1)
    _y = GBVector(y)
    _A = GBMatrix(A)
    _x = GBVector(x)
    y = Ref{Any}()
    time = @belapsed $y[] = spmv_suite_sparse_kernel($_y, $_A, $_x)
    return (;time = time, y = y[])
end

