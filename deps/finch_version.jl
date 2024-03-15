using Pkg

function print_finch_version()
    # Print out where Finch is coming from
    env = Pkg.dependencies()

    if haskey(env[1], "Finch")
        version = env[1]["Finch"].version
        println("Finch Version: $version")

        if haskey(env[1]["Finch"], :dev)
            println("Finch is installed in development mode.")
        else
            println("Finch is installed from the registry.")
        end
    else
        println("Finch is not installed.")
    end
end