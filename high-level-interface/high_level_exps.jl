using Finch
using BenchmarkTools
using MatrixDepot
using DuckDB
using DuckDB.DBInterface
using SparseArrays
using DataStructures
using JSON

include("datasets.jl")
include("duck_db_utils.jl")

function duckdb_triangle_count(A, B, C)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    load_to_duck_db(dbconn, A, ["i", "j"], "A")
    load_to_duck_db(dbconn, B, ["j", "k"], "B")
    load_to_duck_db(dbconn, C, ["i", "k"], "C")
    query_str = "SELECT SUM(A.v * B.v * C.v) as v
                FROM A
                Join B on A.j=B.j
                Join C on C.k=B.k AND C.i=A.i"
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str) seconds=10 samples=3
    return duckdb_time, only(DuckDB.execute(dbconn, query_str))[:v]
end

function finch_triangle(e1, e2, e3)
    return @belapsed begin
        output = Finch.Scalar(0.0)
        @finch begin
            output .= 0
            for j=_, i=_, k=_
                output[] += $e1[i,j] * $e2[k,j] * $e3[k, i]
            end
        end
    end seconds=10 samples=3
end

function hl_triangle(e1, e2, e3)
    hl_time = @belapsed compute(sum(lazy($e1)[:, :, nothing].* lazy($e2)[:, nothing, :] .* lazy($e3)[nothing, :, :]))  seconds=10 samples=3
    return hl_time, compute(sum(lazy(e1)[:, :, nothing].* lazy(e2)[:, nothing, :] .* lazy(e3)[nothing, :, :]))
end

function finch_triangle_dcsc_gallop(e1, e2, e3)
    E1 = Finch.Tensor(SparseList(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(SparseList(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    E3 = Finch.Tensor(SparseList(SparseList(Element(0.0), size(e3)[1]), size(e3)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)
    Finch.copyto!(E3, e3)

    return @belapsed begin
        output = Finch.Scalar(0.0)
        @finch begin
            output .= 0
            for j=_, i=_, k=_
                output[] += $E1[gallop(i),j] * $E2[k,j] * $E3[k, gallop(i)]
            end
        end
    end seconds=10 samples=3
end

function hl_mm_sum(e1, e2)
    hl_time = @belapsed compute(sum(lazy($e1)[:, :, nothing] .* lazy($e2)[nothing, :, :]))  seconds=10 samples=3
    return hl_time, compute(sum(lazy(e1)[:, :, nothing] .* lazy(e2)[nothing, :, :]))
end

function duckdb_mm_sum(A, B)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    load_to_duck_db(dbconn, A, ["i", "j"], "A")
    load_to_duck_db(dbconn, B, ["j", "k"], "B")
    query_str = "SELECT SUM(A.v * B.v) as v
                FROM A
                Join B on A.j=B.j"
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str) seconds=10 samples=3
    return duckdb_time, only(DuckDB.execute(dbconn, query_str))[:v]
end

function finch_mm_sum(e1, e2)
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    return @belapsed begin
        output = Finch.Scalar(0.0)
        @finch begin
            output .= 0
            for j=_, i=_, k=_
                output[] += $E1[i,j] * $E2[k,j]
            end
        end
    end seconds=10 samples=3
end


function hl_mm(e1, e2)
    return @belapsed compute(sum(lazy($e1)[:, :, nothing] .* lazy($e2)[nothing, :, :], dims=[2]))  seconds=10 samples=3
end


function finch_mm(e1, e2)
    E1 = Finch.Tensor(Dense(SparseList(Element(false), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(false), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    return @belapsed begin
        output = Finch.Tensor(Dense(SparseDict(Element(0.0))))
        @finch begin
            output .= 0
            for k=_, j=_, i=_
                output[i, j] += $E1[i, k] * $E2[j, k]
            end
        end
    end seconds=10 samples=3
end

function finch_mm_gustavsons(e1, e2)
    e2 = swizzle(e2, 2, 1)
    E1 = Finch.Tensor(Dense(SparseList(Element(false), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(false), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    return @belapsed begin
        C = Tensor(Dense(SparseList(Element(0.0))))
        w = Tensor(SparseByteMap(Element(0.0)))
        @finch begin
            C .= 0
            for j=_
                w .= 0
                for k=_, i=_; w[i] += $E1[i, follow(k)] * $E2[gallop(k), j] end
                for i=_; C[i, j] = w[i] end
            end
        end
    end seconds=10 samples=3
end

function duckdb_mm(A, B)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    load_to_duck_db(dbconn, A, ["i", "j"], "A")
    load_to_duck_db(dbconn, B, ["j", "k"], "B")
    query_str = "SELECT SUM(A.v * B.v) as v, A.i, B.k
                FROM A
                Join B on A.j=B.j
                GROUP BY A.i, B.k"
    duckdb_result = DuckDB.execute(dbconn, query_str)
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str)  seconds=10 samples=3
    return duckdb_time, duckdb_result
end

function hl_elementwise(A, B, C)
    A = lazy(A)
    B = lazy(B)
    C = lazy(C)
    output = (A .+ B) .* C
    elementwise_time = @belapsed compute($output)
    return elementwise_time, compute(output)
end

function hl_elementwise_unfused(A, B, C)
    elementwise_time = @belapsed A .+ B .* C
    return elementwise_time, A .+ B .* C
end

function duckdb_elementwise(A, B, C)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    load_to_duck_db(dbconn, A, ["i", "j"], "A")
    load_to_duck_db(dbconn, B, ["i", "j"], "B")
    load_to_duck_db(dbconn, C, ["i", "j"], "C")
    query_str = "Select AB.v*C.v, C.i, C.j
                FROM C,
                (SELECT (A.v + B.v) as v, A.i, A.j
                    FROM A
                    Join B on A.j=B.j and A.i = B.i) as AB
                WHERE AB.i = C.i AND AB.j = C.j"

    duckdb_result = DuckDB.execute(dbconn, query_str)
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str)  seconds=10 samples=3
    return duckdb_time, duckdb_result
end

function hl_SDMM(A, B, C)
    hl_time = @belapsed compute(sum(lazy($A)[:, :, nothing].* lazy($B)[nothing, :,  :] .* lazy($C)[:,nothing, :], dims=[2]))  seconds=10 samples=3
    return hl_time, compute(sum(lazy(A)[:, :, nothing].* lazy(B)[nothing, :, :] .* lazy(C)[:,nothing, :], dims=[2]))
end

function duckdb_SDMM(A, B, C)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    load_to_duck_db(dbconn, A, ["i", "j"], "A")
    load_to_duck_db(dbconn, B, ["j", "k"], "B")
    load_to_duck_db(dbconn, C, ["i", "k"], "C")
    query_str = "Select SUM(A.v*B.v*C.v), C.i, C.k
                FROM A, B, C
                WHERE A.j = B.j AND B.k = C.k AND A.i = C.i
                GROUP BY C.i, C.k"

    duckdb_result = DuckDB.execute(dbconn, query_str)
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str)  seconds=10 samples=3
    return duckdb_time, duckdb_result
end

make_entry(time, method, operation, matrix) = OrderedDict("time" => time,
                                                        "method" => method,
                                                        "operation" => operation,
                                                        "matrix" => matrix,
                                                        )

matrices = datasets["yang"]
graph_matrices = ["SNAP/soc-Epinions1", "SNAP/email-EuAll", "SNAP/ca-AstroPh"]
results = []
for matrix in graph_matrices
    main_edge = Tensor(SparseMatrixCSC(matrixdepot(matrix)))
    t_hl, t_hl_count = hl_triangle(main_edge, main_edge, main_edge)
    push!(results, make_entry(t_hl, "Finch", "triangle count", matrix))
    println("t_hl: $(t_hl)")
    t_duckdb, t_duckdb_count = duckdb_triangle_count(main_edge, main_edge, main_edge)
    push!(results, make_entry(t_duckdb, "DuckDB", "triangle count", matrix))
    println(t_hl_count)
    println(t_duckdb_count)
    println(t_hl_count .== t_duckdb_count)
    println("t_duckdb: $(t_duckdb)")

    n,m = size(main_edge)
    l = 100
    A = Tensor(rand(n,l))
    B =  Tensor(rand(l,m))
    sddmm_hl, sddmm_hl_count = hl_SDMM(A, B, main_edge)
    push!(results, make_entry(sddmm_hl, "Finch", "SDDMM", matrix))
    println("sddmm_hl: $(sddmm_hl)")
    sddmm_duckdb, sddmm_duckdb_count = duckdb_SDMM(A, B, main_edge)
    push!(results, make_entry(sddmm_duckdb, "DuckDB", "SDDMM", matrix))
    println("sddmm_duckdb: $(sddmm_duckdb)")

    mm_hl = hl_mm(main_edge, main_edge)
    push!(results, make_entry(mm_hl, "Finch", "AA'", matrix))
    println("mm_hl: $(mm_hl)")
    mm_duckdb, mm_duckdb_result = duckdb_mm(main_edge, main_edge)
    push!(results, make_entry(mm_duckdb, "DuckDB", "AA'", matrix))
    println("mm_duckdb: $(mm_duckdb)")
end

elementwise_matrices = [("DIMACS10/smallworld", "DIMACS10/preferentialAttachment", "DIMACS10/G_n_pin_pout")]

for (A,B,C) in elementwise_matrices
    A = Tensor(SparseMatrixCSC(matrixdepot(A)))
    B = Tensor(SparseMatrixCSC(matrixdepot(B)))
    C = Tensor(SparseMatrixCSC(matrixdepot(C)))

    element_hl, element_hl_count = hl_elementwise(A, B, C)
    push!(results, make_entry(element_hl, "Finch", "(A.+B).* C", "($A,$B,$C)"))
    println("element_hl: $(element_hl)")
    element_duckdb, element_duckdb_result = duckdb_elementwise(A, B, C)
    push!(results, make_entry(element_duckdb, "DuckDB", "(A.+B).* C",  "($A,$B,$C)"))
    println("element_duckdb: $(element_duckdb)")
end

write("results.json", JSON.json(results, 4))
