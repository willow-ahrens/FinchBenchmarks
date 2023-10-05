using TensorDepot
using Documenter

DocMeta.setdocmeta!(TensorDepot, :DocTestSetup, :(using TensorDepot); recursive=true)

makedocs(;
    modules=[TensorDepot],
    authors="Willow Ahrens",
    repo="https://github.com/peterahrens/TensorDepot.jl/blob/{commit}{path}#{line}",
    sitename="TensorDepot.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://peterahrens.github.io/TensorDepot.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/peterahrens/TensorDepot.jl",
    devbranch="main",
)
