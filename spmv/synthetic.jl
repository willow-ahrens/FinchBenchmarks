using Random
using LinearAlgebra

function random_permutation_matrix(n)
    perm = randperm(n) 
    P = I(n) 
    return P[perm, :]
end

function reverse_permutation_matrix(n)
    return [i == j ? 0 : (i+j == n+1 ? 1 : 0) for i in 1:n, j in 1:n]
end

function banded_matrix(n, b)
    banded = zeros(n, n)
    for i in 1:n
        for j in max(1, i - b):min(n, i + b)
            banded[i, j] = i + j  # Adjust values as needed
        end
    end
    return banded
end