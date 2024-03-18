using MatrixDepot
using SparseArrays

matrix_names = [
    "FEMLAB/poisson3Da",
    "SNAP/wiki-Vote",
    "SNAP/email-Enron",
    "SNAP/ca-CondMat",
    "Oberwolfach/filter3D",
    "Williams/cop20k_A",
    "Um/offshore",
    "Um/2cubes_sphere",
    "vanHeukelum/cage12",
    "SNAP/amazon0312",
    "Hamm/scircuit",
    "SNAP/web-Google",
    "GHS_indef/mario002",
    "SNAP/cit-Patents",
    "JGD_Homology/m133-b3",
    "Williams/webbase-1M",
    "SNAP/roadNet-CA",
    "SNAP/p2p-Gnutella31",
    "Pajek/patents_main"
]

matrix_names = [
    "SNAP/email-Eu-core",
    "SNAP/CollegeMsg",
    "SNAP/soc-sign-bitcoin-alpha",
    "SNAP/ca-GrQc",
    "SNAP/soc-sign-bitcoin-otc",
    "SNAP/p2p-Gnutella08",
    "SNAP/as-735",
    "SNAP/p2p-Gnutella09",
    "SNAP/wiki-Vote",
    "SNAP/p2p-Gnutella06",
    "SNAP/p2p-Gnutella05",
    "SNAP/ca-HepTh"
]

matrices_with_info = []
for name in matrix_names
    md = SparseMatrixCSC(matrixdepot(name))
    rows, cols = size(md)
    nonzeros = nnz(md)
    push!(matrices_with_info, Dict(
        "name" => name,
        "rows" => rows,
        "cols" => cols,
        "nonzeros" => nonzeros
    ))
end

sort!(matrices_with_info, by = x -> x["rows"])

for m in matrices_with_info
    println("$(m["name"]): $(m["rows"])x$(m["cols"]) ($(m["nonzeros"]))")
end
