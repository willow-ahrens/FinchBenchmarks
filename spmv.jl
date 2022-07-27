using Finch
using SparseArrays

include("TensorMarket.jl")
using .TensorMarket

#=
function readtns_fsparse(fname)
    I = nothing
    V = Float64[]
    for line in readlines(fname)
        if length(line) > 1
            line = split(line, "#")[1]
            entries = split(line)
            if length(entries) >= 1
                if isnothing(I)
                    I = ((Int[] for _ in 1:length(entries) - 1)...,)
                end
                for (n, e) in enumerate(entries[1:end-1])
                    push!(I[n], parse(Int, e))
                end
                push!(V, parse(Float64, entries[end]))
            end
        end
    end
    return (I, V)
end

function writetns_fsparse(fname, I, V)
    open(fname, "w") do io
        for (crd, val) in zip(zip(I...), V)
            write(io, join(crd, " "))
            write(io, " ")
            write(io, string(val))
            write(io, "\n")
        end
    end
end
=#

function spmv_taco(A, x)
    y_ref = fiber(x)
    @index @loop i j y_ref[i] += A[i, j] * x[j]
    @index @loop i y_ref[i] = 0

    ttwrite("A.ttx", ffindnz(A)...)
    ttwrite("x.ttx", ffindnz(x)...)
    ttwrite("y.ttx", ffindnz(y_ref)...)

    io = IOBuffer()

    run(pipeline(`./spmv_taco y.ttx A.ttx x.ttx`, stdout=io))

    y = fsparse(ttread("y.ttx")...)

    @assert FiberArray(y) == FiberArray(y_ref)

    return parse(Int64, String(take!(io))) * 1.0e-9
end

spmv_taco(ones(10, 10), ones(10))