using Pkg

function print_diagnostics()
    Pkg.status("Finch")

    println("Julia Version: $(VERSION)")
end