using Finch
using BenchmarkTools
using MatrixDepot
using DuckDB
using DuckDB.DBInterface

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
    println("DuckDB Triangle Output: ", DuckDB.execute(dbconn, query_str))
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str)
    return duckdb_time
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
    end
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
    end
end


function finch_triangle_gallop(e1, e2, e3)
    return @belapsed begin
        output = Finch.Scalar(0.0)
        @finch begin
            output .= 0
            for j=_, i=_, k=_
                output[] += $e1[gallop(i), j] * $e2[gallop(k), j] * $e3[gallop(k), i]
            end
        end
    end
end


function duckdb_mm_sum(A, B)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    load_to_duck_db(dbconn, A, ["i", "j"], "A")
    load_to_duck_db(dbconn, B, ["j", "k"], "B")
    query_str = "SELECT SUM(A.v * B.v) as v
                FROM A
                Join B on A.j=B.j"
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str)
    return duckdb_time
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
    end
end


function finch_mm_sum_mat(e1, e2)
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    return @belapsed begin
        I = Tensor(Dense(Element(0.0)))
        output = Finch.Scalar(0.0)
        @finch begin
            I .= 0
            for j=_, i=_
                I[j] += $E1[i,j]
            end
        end
        @finch begin
            output .= 0
            for j=_, k=_
                output[] += I[j] * $E2[k,j]
            end
        end
    end
end


function finch_mm_dcsc(e1, e2)
    E1 = Finch.Tensor(SparseList(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(SparseList(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    return @belapsed begin
        output = Finch.Tensor(SparseHash{1}(SparseHash{1}(Element(0.0))))
        @finch begin
            output .= 0
            for k=_, j=_, i=_
                output[i, j] += $E1[i, k] * $E2[j, k]
            end
        end
    end
end


function finch_mm(e1, e2)
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    return @belapsed begin
        output = Finch.Tensor(Dense(SparseHash{1}(Element(0.0))))
        @finch begin
            output .= 0
            for k=_, j=_, i=_
                output[i, j] += $E1[i, k] * $E2[j, k]
            end
        end
    end
end

function finch_mm_gustavsons(e1, e2)
    e2 = swizzle(e2, 2, 1)
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
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
    end
end

function finch_mm_proper_row_major(e1, e2)
    e2 = swizzle(e2, 2, 1)
    E1 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e1)[1]), size(e1)[2]))
    E2 = Finch.Tensor(Dense(SparseList(Element(0.0), size(e2)[1]), size(e2)[2]))
    Finch.copyto!(E1, e1)
    Finch.copyto!(E2, e2)

    return @belapsed begin
        C = Tensor(Dense(SparseHash{1}(Element(0.0))))
        w = Tensor(SparseByteMap(Element(0.0)))
        @finch begin
            C .= 0
            for j=_, k=_, i=_;
                C[i, j] += $E1[i, k] * $E2[k, j]
            end
        end
    end
end

function finch_mm_proper_inner(e1, e2)
    return @belapsed begin
        z = Finch.default(e1) * Finch.default(e2) + false
        C = Tensor(Dense(SparseList(Element(z))))
        w = Tensor(SparseDict(SparseDict(Element(z))))
        AT = Tensor(Dense(SparseList(Element(z))))
        @finch mode=fastfinch (w .= 0; for k=_, i=_; w[k, i] = $e1[i, k] end)
        @finch mode=fastfinch (AT .= 0; for i=_, k=_; AT[k, i] = w[k, i] end)
        @finch (C .= 0; for j=_, i=_, k=_; C[i, j] += AT[k, gallop(i)] * $e2[k, gallop(j)] end)
    end
end

function duckdb_mm(A, B)
    dbconn = DBInterface.connect(DuckDB.DB, ":memory:")
    load_to_duck_db(dbconn, A, ["i", "j"], "A")
    load_to_duck_db(dbconn, B, ["j", "k"], "B")
    query_str = "SELECT SUM(A.v * B.v) as v, A.i, B.k
                FROM A
                Join B on A.j=B.j
                GROUP BY A.i, B.k"
    duckdb_time = @belapsed DuckDB.execute($dbconn, $query_str)
    return duckdb_time
end

main_edge = Tensor(matrixdepot("SNAP/soc-Epinions1"))
t_duckdb = duckdb_triangle_count(main_edge, main_edge, main_edge)
t_finch = finch_triangle(main_edge, main_edge, main_edge)
t_finch = finch_triangle(main_edge, main_edge, main_edge)
t_finch_gallop = finch_triangle_gallop(main_edge, main_edge, main_edge)
t_finch_gallop = finch_triangle_gallop(main_edge, main_edge, main_edge)
t_finch_dcsc_gallop = finch_triangle_dcsc_gallop(main_edge, main_edge, main_edge)
t_finch_dcsc_gallop = finch_triangle_dcsc_gallop(main_edge, main_edge, main_edge)
println("t_duckdb: $(t_duckdb)")
println("t_finch: $(t_finch)")
println("t_finch_gallop: $(t_finch_gallop)")
println("t_finch_dcsc_gallop: $(t_finch_dcsc_gallop)")

mm_duckdb = duckdb_mm_sum(main_edge, main_edge)
mm_finch = finch_mm_sum(main_edge, main_edge)
mm_finch = finch_mm_sum(main_edge, main_edge)
mm_finch_materialize = finch_mm_sum_mat(main_edge, main_edge)
mm_finch_materialize = finch_mm_sum_mat(main_edge, main_edge)
println("mmsum_finch: $(mm_finch)")
println("mmsum_finch_materialize: $(mm_finch_materialize)")

mm_duckdb = duckdb_mm(main_edge, main_edge)
mm_finch = finch_mm(main_edge, main_edge)
mm_finch = finch_mm(main_edge, main_edge)
mm_finch_gustavsons = finch_mm_gustavsons(main_edge, main_edge)
mm_finch_gustavsons = finch_mm_gustavsons(main_edge, main_edge)
mm_finch_dcsc = finch_mm_dcsc(main_edge, main_edge)
mm_finch_dcsc = finch_mm_dcsc(main_edge, main_edge)
println("mm_duckdb: $(mm_duckdb)")
println("mm_finch: $(mm_finch)")
println("mm_finch_gustavsons: $(mm_finch_gustavsons)")
println("mm_finch_dcsc: $(mm_finch_dcsc)")
