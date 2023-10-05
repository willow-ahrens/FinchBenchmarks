using MatrixDepot

mtxs = listdata("Boeing/*")
sort!(mtxs, by=(mtx->mtx.nnz))
for mtx in (map(mtx -> mtx.name, mtxs))
    println("(\"$mtx\", \"$(mtx[8:end])\"),")
end