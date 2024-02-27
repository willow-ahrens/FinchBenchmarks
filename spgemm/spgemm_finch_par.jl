using Finch
using BenchmarkTools
using Base.Threads


function spgemm_finch_gustavson_kernel_parallel(A, B)
    # @assert Threads.nthreads() >= 2
    z = default(A) * default(B) + false
    C = Tensor(Dense(Seperation(SparseList(Element(z)))))
    w = moveto(Tensor(Dense(Element(z))), CPULocalMemory(CPU()))
    @finch_code begin
        C .= 0
        for j=parallel(_)
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
    end
    @finch begin
        C .= 0
        for j=parallel(_)
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
    end
    return C
end


function spgemm_finch_parallel(f, A, B)
    _A = Tensor(A)
    _B = Tensor(B)
    C = Ref{Any}()
    time = @belapsed $C[] = $f($_A, $_B)
    return (;time = time, C = C[])
end


spgemm_finch_gustavson_parallel(A, B) = spgemm_finch_parallel(spgemm_finch_gustavson_kernel_parallel, A, B)
