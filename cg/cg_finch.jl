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
                        if i <= j
                            y[i] += A_ij * x_j
                        end
                        if i < j
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

# Using algorithm from https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/e5d10e22552916e7dbd73d0a9ece02d873dee375/src/cg.jl
function cg_finch_kernel(x, A, b, l)
    (n, m) = size(A)
    @assert n == m

    _x = Tensor(Dense(Element(0.0)), undef, n)
    u = Tensor(Dense(Element(0.0)), undef, n)
    _u = Tensor(Dense(Element(0.0)), undef, n)
    r = Tensor(Dense(Element(0.0)), undef, n)
    _r = Tensor(Dense(Element(0.0)), undef, n)
    c = Tensor(Dense(Element(0.0)), undef, n)

    ssymv_finch(c, A, x)
    @finch for i = _; r[i] = b[i] - c[i] end
    residual_sq = Scalar(0.0)
    @finch (for i = _; residual_sq[] += r[i]^2 end)
    residual = Scalar((residual_sq[])^(1/2))

    prev_residual = Scalar(1.0)
    for k = 1:l
        β = Scalar(residual[]^2 / prev_residual[]^2)
        @finch (for i = _; _u[i] = r[i] + β[] * u[i] end)
        u = _u
        
        c = ssymv_finch(c, A, u)
        uc = Scalar(0.0)
        @finch (for i = _; uc[] += u[i] * c[i] end)
        α = Scalar(residual[]^2 / uc[])


        @finch (for i = _; _x[i] = x[i] + α[] * u[i] end)
        @finch (for i = _; _r[i] = r[i] - α[] * c[i] end)
        x = _x
        r = _r
        
        prev_residual = residual[]
        residual_sq = Scalar(0.0)
        @finch (for i = _; residual_sq[] += r[i]^2 end)
        residual = Scalar((residual_sq[])^(1/2))
    end
    x
end

function cg_finch(x, A, b, l)
    _x = Tensor(Dense(Element(0.0)), x)
    _A = Tensor(Dense(SparseList(Element(0.0))), A)
    _b = Tensor(Dense(Element(0.0)), b)
    x = Ref{Any}()
    time = @belapsed $x[] = cg_finch_kernel($_x, $_A, $_b, $l)
    return (;time = time, x = x[])
end