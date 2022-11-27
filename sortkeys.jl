using JSON
function main(input, output, keys...)
    data = JSON.parse(open(input))
    open(output, "w") do f
        println(f, "[")
        comma = false
        for d in data
            if comma
                println(f, ",")
            end
            println(f, "    {")
            comma2 = false
            for arg in keys
                if comma2
                    println(f, ",")
                end
                print(f, "        \"$arg\": $(repr(d[arg]))")
                comma2=true
            end
            println(f)
            print(f, "    }")
            comma = true
        end
        println(f)
        println(f, "]")
    end
end
main(ARGS...)   