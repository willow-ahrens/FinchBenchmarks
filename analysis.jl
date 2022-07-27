using UnicodePlots
using BSON
using JSON
using Statistics
using PrettyTables

function main(args)
    println("figure 6:")
    figure6 = open("figure6.txt", "w")
    header = (
        ["Kernel", "Min-Depth", "Undominated", "Min-Depth", "Undominated", "Asymptotic"],
        ["",       "Schedules", "Schedules",   "Schedules", "Schedules",   "Filter"],
        ["",       "",          "",            "(TACO)",    "(TACO)",      "Runtime"],
    )
    table = Array{Any}(undef, 0, 6)
    for arg in args
        data = Dict()
        open(arg, "r") do f
            data = JSON.parse(f)
        end
        name = basename(first(splitext(arg)))
        row = [
            name,
            get(data, "universe_length", "TIMEOUT"),
            get(data, "frontier_length", "TIMEOUT"),
            data["tacoverse_length"],
            data["tacotier_length"],
            data["tacotier_filter_time"] / data["tacoverse_length"],
        ]
        table = vcat(table, permutedims(row))
    end
    pretty_table(table, header=header)
    pretty_table(figure6, table, header=header)
    close(figure6)

    println("figure 7:")
    figure7 = open("figure7.txt", "w")
    for arg in args
        data = Dict()
        open(arg, "r") do f
            data = JSON.parse(f)
        end
        name = basename(first(splitext(arg)))

        function relim((x, y))
            c = (x + y)/2
            dx, dy = (x, y) .- c
            return (dx, dy) .* 1.2 .+ c
        end
            

        println("$name:\n")
        xlim = relim(log10.(extrema(data["n_series"])))
        ylim = relim(log10.(extrema(vcat(data["default_n_series"], data["auto_n_series"]))))
        p = lineplot(log10.(data["n_series"]), log10.(data["default_n_series"]), color=:blue, title="Runtime vs. Dimension (p=0.01)", xlabel="Log10 Dimension n", ylabel="Log10 Runtime (Seconds)", xlim=xlim, ylim=ylim)
        p = scatterplot!(p, log10.(data["n_series"]), log10.(data["default_n_series"]), name="Default Schedule (+)", color=:blue, marker=:+)
        p = lineplot!(p, log10.(data["n_series"]), log10.(data["auto_n_series"]), color=:red)
        p = scatterplot!(p, log10.(data["n_series"]), log10.(data["auto_n_series"]), name="Tuned Schedule (O)", color=:red, marker=:O)
        println(p)
        println()

        xlim = relim(log10.(extrema(data["p_series"])))
        ylim = relim(log10.(extrema(vcat(data["default_p_series"], data["auto_p_series"]))))
        p = lineplot(log10.(data["p_series"]), log10.(data["default_p_series"]), color=:blue, title="Runtime vs. Density (n=$(data["N"]))", xlabel="Log10 Density p", ylabel="Log10 Runtime (Seconds)", xlim=xlim, ylim=ylim)
        p = scatterplot!(p, log10.(data["p_series"]), log10.(data["default_p_series"]), name="Default Schedule (+)", color=:blue, marker=:+)
        p = lineplot!(p, log10.(data["p_series"]), log10.(data["auto_p_series"]), color=:red)
        p = scatterplot!(p, log10.(data["p_series"]), log10.(data["auto_p_series"]), name="Tuned Schedule (O)", color=:red, marker=:O)
        println(p)
        println()

        println(figure7, "$name:\n")
        pretty_table(figure7,
            [[["Default Runtime (seconds)"] data["default_n_series"]'];
                [["Tuned Runtime (seconds)"] data["auto_n_series"]'];
                [["Speedup"] (data["default_n_series"]./data["auto_n_series"])']],
            header = [["Dimension (p = 0.01)"]; data["n_series"]])
        println(figure7, "\n")

        pretty_table(figure7,
            [[["Default Runtime (seconds)"] data["default_p_series"]'];
                [["Tuned Runtime (seconds)"] data["auto_p_series"]'];
                [["Speedup"] (data["default_p_series"]./data["auto_p_series"])']],
            header = [["Density (n = $(data["N"]))"]; data["p_series"]])
        println(figure7, "\n")
    end
    close(figure7)
end

main(ARGS)
