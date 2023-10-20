using Finch
using BenchmarkTools

function ssymv_finch(y, A, x)
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

function cg_finch_kernel(x, A, b, l)
    (n, m) = size(A)
    @assert n == m

    r = Fiber!(Dense(Element(0.0), n))
    _r = Fiber!(Dense(Element(0.0), n))
    p = Fiber!(Dense(Element(0.0), n))
    _p = Fiber!(Dense(Element(0.0), n))
    _x = Fiber!(Dense(Element(0.0), n))
    Ap = Fiber!(Dense(Element(0.0), n))

    # r_0 = b - Ax_0
    ssymv_finch(_r, A, x)
    @finch for i = _; r[i] = b[i] - _r[i] end
    # p_0 = r_0
    @finch for i = _; p[i] = r[i] end
    # r2 = r_0^Tr_0
    r2 = Scalar(0.0)
    @finch (for i = _; r2[] += r[i]^2 end)

    for k = 1:l
        pTAp = Scalar(0.0)

        Ap = ssymv_finch(Ap, A, p)

        # alpha_k = r_k^T * r_k / (p_k^T * A * p_k)
        @finch (pTAp .= 0; for i = _; pTAp[] += p[i] * Ap[i] end)

        alpha = Scalar(r2[] / pTAp[])

        @finch (for i = _; _x[i] = x[i] + alpha[] * p[i] end)
        @finch (for i = _; _r[i] = r[i] - alpha[] * Ap[i] end)

        _r2 = Scalar(0.0)
        @finch (for i = _; _r2[] += _r[i]^2 end)

        # beta_k = r_k+1^T * r_k+1 / (r_k^T * r_k)
        beta = Scalar(_r2[] / r2[])

        # p_k+1 = r_k+1 * beta_k * p_k
        @finch (for i = _; _p[i] = _r[i] + beta[] * p[i] end)

        x = _x
        r = _r
        p = _p
        r2 = _r2
    end
    (x,)
end

function cg_finch(x, A, b, l)
    _x = Fiber!(Dense(Element(0.0)), x)
    _A = Fiber!(Dense(SparseList(Element(0.0))), A)
    _b = Fiber!(Dense(Element(0.0)), b)
    x = Ref{Any}()
    time = @belapsed $x[] = cg_finch_kernel($_x, $_A, $_b, $l)
    return (;time = time, x = x[])
end