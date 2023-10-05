if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using Finch
using MatrixDepot
using TensorMarket
using SparseArrays
using NPZ
using HDF5

generator = Dict()

"""
    generate(problem, name, dst, ext, args...)

Generate a problem of type `problem` with generator named `name` and save it to
the path `dst` with extension `ext`.
"""
function generate(problem, name, dst, ext, args...)
    generator[problem][name](dst, lstrip(ext, '.'), args...)
end

function spmv(dst, ext, mtx)
    A = SparseMatrixCSC(matrixdepot(mtx))
    m, n = size(A)
    x = rand(n)
    y = A * x
    Finch.fwrite(joinpath(dst, "y_ref.$ext"), copyto!(Fiber!(Dense(Element(0.0))), y))
    Finch.fwrite(joinpath(dst, "A.$ext"), copyto!(swizzle(Fiber!(Dense(SparseList(Element(0.0)))), 2, 1), A))
    Finch.fwrite(joinpath(dst, "x.$ext"), copyto!(Fiber!(Dense(Element(0.0))), x))
end 

generator["spmv"] = Dict()
generator["spmv"]["matrixmarket"] = spmv

function main(args...)
    generate(args...)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS...)
end