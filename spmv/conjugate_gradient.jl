using Finch
using BenchmarkTools
using IterativeSolvers
using SparseArrays

n = 1000
sparsity = 0.1
A = Fiber!(Dense(SparseList(Element(0.0))), fsprand((n, n), sparsity))
b = Fiber!(Dense(Element(0.0)), rand(n))
x = Fiber!(Dense(Element(0.0)), zeros(n))
y = Fiber!(Dense(Element(0.0)), zeros(n))

temp2 = Scalar(0.0)
eval(@finch_kernel function spmv(y, A, x, temp2)
    for j = _
        let temp1 = x[j]
            temp2 .= 0
            for i = _
                let temp3 = A[i, j]
                    if uptrimask[i, j]
                        y[i] += temp1 * temp3
                    end
                    if uptrimask[i, j - 1]
                        temp2[] += temp3 * x[i]
                    end
                end
            end
            y[j] += temp2[]
        end
    end
end)

function conjugate_gradient(A, x, b)
    (n, m) = size(A)
    @assert n == m

    temp = Fiber!(Dense(Element(0.0)), zeros(n))
    r = Fiber!(Dense(Element(0.0)), zeros(n))
    _r = Fiber!(Dense(Element(0.0)), zeros(n))
    p = Fiber!(Dense(Element(0.0)), zeros(n))
    _p = Fiber!(Dense(Element(0.0)), zeros(n))
    _x = Fiber!(Dense(Element(0.0)), zeros(n))

    spmv(temp, A, x, temp2)
    # r_0 = b - Ax_0
    @finch for i = _ r[i] = b[i] - temp[i] end
    # p_0 = r_0
    @finch for i = _ p[i] = r[i] end

    for k = 1:100 
        num = Scalar(0.0)
        _num = Scalar(0.0)
        den = Scalar(0.0)
        alpha = Scalar(0.0)
        beta = Scalar(0.0)

        t = Fiber!(Dense(Element(0.0)), zeros(n))
        spmv(t, A, p, temp2)
        @finch begin
            for i = _ 
                num[] += r[i]^2
                den[] += p[i] * t[i]
            end
        end

        @finch begin
            # alpha_k = r_k^T * r_k / (p_k^T * A * p_k)
            alpha[] = num[] / den[]
            # x_k+1 = x_k + alpha_k * p_k
            # r_k+1 = r_k - alpha_k * A * p_k
            for i = _
                _x[i] = x[i] + alpha[] * p[i]
                _r[i] = r[i] - alpha[] * t[i]
            end

            for i = _
                _num[] += _r[i]^2
            end
            # beta_k = r_k+1^T * r_k+1 / (r_k^T * r_k)
            beta[] = _num[] / num[]

            # p_k+1 = r_k+1 * beta_k * p_k
            for i = _
                _p[i] = _r[i] + beta[] * p[i]
            end

            for i = _
                x[i] = _x[i]
                r[i] = _r[i]
                p[i] = _p[i]
            end
        end
    end
end

AA = sprand(n, n, 0.1)
xx = zeros(n)
bb = rand(n)
function iterative_solvers_cg(AA, xx, bb)
    cg!(xx, AA, bb; maxiter = 100)
end

time_finch = @belapsed conjugate_gradient(A, x, b)
time_iterative_solvers = @belapsed iterative_solvers_cg(AA, xx, bb)
println("Finch time: ", time_finch)
println("IterativeSolvers time: ", time_iterative_solvers)