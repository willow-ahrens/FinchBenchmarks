using Random
using LinearAlgebra

function random_permutation_matrix(n)
    perm = randperm(n) 
    P = I(n) 
    return P[perm, :]
end

function reverse_permutation_matrix(n)
    perm = reverse([i for i in 1:n])
    P = I(n) 
    return P[perm, :]
end

function banded_matrix(n, b)
    banded = zeros(n, n)
    for i in 1:n
        for j in max(1, i - b):min(n, i + b)
            banded[i, j] = j - i + 1 
        end
    end
    return banded
end

function blocked_matrix(n, b)
    return kron(sprand(n, n, 0.01), ones(b, b))
end