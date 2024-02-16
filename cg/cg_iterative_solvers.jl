using IterativeSolvers

function cg_iterative_solvers_kernel(x, A, b, l)
    IterativeSolvers.cg(A, b; abstol = zero(eltype(A)), reltol = zero(eltype(A)), maxiter = l, log = false)
end

function cg_iterative_solvers(x, A, b, l)
    _x = Vector(x)
    _A = SparseMatrixCSC(A)
    _b = Vector(b)
    x = Ref{Any}()
    time = @belapsed $x[] = cg_iterative_solvers_kernel($_x, $_A, $_b, $l)
    return (;time = time, x = x[])
end