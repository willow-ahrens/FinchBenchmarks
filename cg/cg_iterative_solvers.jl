using IterativeSolvers

function cg_iterative_solvers_format(x, A, b)
    x = Vector(x)
    A = Symmetric(SparseMatrixCSC(A))
    b = Vector(b)
    (b, A, x)
end

function cg_iterative_solvers(x, A, b, l)
    IterativeSolvers.cg!(x, A, b; abstol = zero(eltype(A)), reltol = zero(eltype(A)), maxiter = l, log = false)
    (x,)
end