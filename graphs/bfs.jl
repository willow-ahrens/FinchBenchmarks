#pull is row-major
#push is col-major

V = Tensor(Dense(Element(false)))
P = Tensor(Dense(Element(0)))
F = Tensor(SparseByteMap(Pattern()))
_F = Tensor(SparseByteMap(Pattern()))
p = ShortCircuitScalar{0}()
A = Tensor(Dense(SparseList(Pattern())))
AT = Tensor(Dense(SparseList(Pattern())))

eval(@finch_kernel function finch_bfs_push_kernel(_F, F, A, V, P)
    _F .= false
    for j=_, k=_
        if F[j] && A[k, j] && !(V[k])
            _F[k] |= true
            P[k] <<choose(0)>>= j #Only set the parent for this vertex
        end
    end
    return _F
end)
function finch_bfs_push(_F, F, A, V, P)
    return finch_bfs_push_kernel(_F, F, A, V, P)
end

eval(@finch_kernel function finch_bfs_pull_kernel(_F, F, AT, V, P, p)
    _F .= false
    for k=_
        if !V[k]
            p .= 0
            for j=_
                if F[follow(j)] && AT[j, k]
                    p[] <<choose(0)>>= j #Only set the parent for this vertex
                end
            end
            if p[] != 0
                _F[k] |= true
                P[k] = p[]
            end
        end
    end
    return _F
end)

function finch_bfs_pull(_F, F, AT, V, P)
    p = ShortCircuitScalar{0}()
    return finch_bfs_pull_kernel(_F, F, AT, V, P, p)
end

"""
    bfs(edges; [source])

Calculate a breadth-first search on the graph specified by the `edges` adjacency
matrix. Return the node numbering.
"""
function bfs_finch_kernel(edges, edgesT, source=5, alpha = 0.01)
    (n, m) = size(edges)
    edges = pattern!(edges)

    @assert n == m
    F = Tensor(SparseByteMap(Pattern()), n)
    _F = Tensor(SparseByteMap(Pattern()), n)
    @finch F[source] = true
    F_nnz = 1

    V = Tensor(Dense(Element(false)), n)
    @finch V[source] = true

    P = Tensor(Dense(Element(0)), n)
    @finch P[source] = source

    while F_nnz > 0
        if F_nnz/m > alpha
            finch_bfs_pull(_F, F, edgesT, V, P)
        else
            finch_bfs_push(_F, F, edges, V, P)
        end
        c = Scalar(0)
        @finch begin
            for k=_
                let _f = _F[k]
                    V[k] |= _f
                    c[] += _f
                end
            end
        end
        (F, _F) = (_F, F)
        F_nnz = c[]
    end
    return P
end