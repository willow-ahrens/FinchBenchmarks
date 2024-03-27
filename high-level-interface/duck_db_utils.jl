

function tensor_to_vec_of_tuples(t::Tensor)
    return collect(zip(ffindnz(t)...))
end

function create_table(dbconn, idx_names, table_name)
    create_str = "CREATE OR REPLACE TABLE $table_name ("
    prefix = ""
    for idx in idx_names
        create_str *= "$prefix $(idx) INT64"
        prefix = ", "
    end
    create_str *= "$prefix v DOUBLE)"
    DBInterface.execute(dbconn, create_str)
end

function load_to_duck_db(dbconn, tensor, idx_names, table_name)
    create_table(dbconn, idx_names, table_name)
    appender = DuckDB.Appender(dbconn, "$table_name")
    data = tensor_to_vec_of_tuples(tensor)
    for row in data
        for val in row
            DuckDB.append(appender, val)
        end
        DuckDB.end_row(appender)
    end
    DuckDB.flush(appender)
    DuckDB.close(appender)
end
