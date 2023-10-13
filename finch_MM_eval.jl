using Finch
using BenchmarkTools
using JSON
using MatrixDepot
using SparseArrays

function spgemm_inner(A, B)
    z = default(A) * default(B) + false
    C = Fiber!(Dense(SparseList(Element(z))))
    w = Fiber!(SparseHash{2}(Element(z)))
    AT = Fiber!(Dense(SparseList(Element(z))))
    @finch mode=fastfinch (w .= 0; for k=_, i=_; w[k, i] = A[i, k] end)
    @finch mode=fastfinch (AT .= 0; for i=_, k=_; AT[k, i] = w[k, i] end)
    @finch (C .= 0; for j=_, i=_, k=_; C[i, j] += AT[k, i] * B[k, j] end)
    return C
end

function spgemm_outer(A, B)
    z = default(A) * default(B) + false
    C = Fiber!(Dense(SparseList(Element(z))))
    w = Fiber!(SparseHash{2}(Element(z)))
    BT = Fiber!(Dense(SparseList(Element(z))))
    @finch mode=fastfinch (w .= 0; for j=_, k=_; w[j, k] = B[k, j] end)
    @finch (BT .= 0; for k=_, j=_; BT[j, k] = w[j, k] end)
    @finch (w .= 0; for k=_, j=_, i=_; w[i, j] += A[i, k] * BT[j, k] end)
    @finch (C .= 0; for j=_, i=_; C[i, j] = w[i, j] end)
    return C
end

function spgemm_gustavson(A, B)
    z = default(A) * default(B) + false
    C = Fiber!(Dense(SparseList(Element(z))))
    w = Fiber!(SparseByteMap(Element(z)))
    @finch begin
        C .= 0
        for j=_
            w .= 0
            for k=_, i=_; w[i] += A[i, k] * B[k, j] end
            for i=_; C[i, j] = w[i] end
        end
    end
    return C
end

function main(resultfile)
        #for mtx in ["HB/bcspwr07"]
        for mtx in ["FEMLAB/poisson3Da", "Oberwolfach/filter3D", "Williams/cop20k_A", "Um/offshore", "Um/2cubes_sphere", "vanHeukelum/cage12", "SNAP/wiki-Vote", "SNAP/email-Enron", "SNAP/ca-CondMat", "SNAP/amazon0312", "Hamm/scircuit", "SNAP/web-Google", "GHS_indef/mario002", "SNAP/cit-Patents", "JGD_Homology/m133-b3", "Williams/webbase-1M", "SNAP/roadNet-CA", "SNAP/p2p-Gnutella31","Pajek/patents_main"]
                A = fiber(SparseMatrixCSC(matrixdepot(mtx)))
                time_g = @belapsed spgemm_gustavson($A, $A) evals=1
                time_i = @belapsed spgemm_inner($A, $A) evals=1
                time_o = @belapsed spgemm_outer($A, $A) evals=1
                result = Dict("matrix"=>mtx, "gustavson"=>time_g, "inner"=>time_i, "outer"=>time_o)
                open(resultfile,"w") do f
                        JSON.print(f, JSON.json(result))
                end
        end
end
