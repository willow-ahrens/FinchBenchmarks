for kernel in Serialization.deserialize(joinpath(@__DIR__, "fill_kernels.jls"))
    eval(kernel)
end

function fill_opencv_kernel(mask, filter, x, y)
    seed_point = OpenCV.Point(Int32(x - 1), Int32(y - 1))
    (c, h, w) = size(mask)
    flood_mask = zeros(UInt8, 1, h + 2, w + 2)
    OpenCV.floodFill(copy(mask), flood_mask, seed_point, (0x02,))
    return flood_mask[1, 2:end-1, 2:end-1]
end

function fill_opencv((img, x, y),)
    mask = Array{UInt8}(reshape(img .!= 0, 1, size(img)...))
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed fill_opencv_kernel($mask, $filter, $x, $y) evals=1
    output = Array(fill_opencv_kernel(mask, filter, x, y))
    return (; time = time, mem = summarysize(img), nnz = length(img), output = output)
end

function fill_finch_kernel(frontier_2, frontier, mask, image, tmp, x, y)
    @finch begin
        mask .= false
        mask[x, y] = true
        frontier .= false
        frontier[x, y] = true
    end
    frontier_2 = fill_finch_step_kernel(frontier_2, frontier, mask, image, tmp).frontier_2
    while countstored(frontier_2) > 0
        (frontier_2, frontier) = (frontier, frontier_2)
        frontier_2 = fill_finch_step_kernel(frontier_2, frontier, mask, image, tmp).frontier_2
    end
    return mask
end

function fill_finch((img, x, y),)
    (xs, ys) = size(img)

    frontier_2 = Tensor(SparseList(SparseList(Pattern())), undef, xs, ys)
    frontier = Tensor(SparseList(SparseList(Pattern())), undef, xs, ys)
    mask = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    image = Tensor(Dense(Dense(Element(false))), img)
    tmp = Tensor(SparseList(SparseList(Pattern())))
    time = @belapsed fill_finch_kernel($frontier_2, $frontier, $mask, $image, $tmp, $x, $y)
    return (;time=time, mem = summarysize(mask), nnz = countstored(mask), output=mask)
end

function fill_finch_scatter_kernel(frontier_2, frontier, mask, image, x, y)
    @finch begin
        mask .= false
        mask[x, y] = true
        frontier .= false
        frontier[x, y] = true
    end
    res = fill_finch_scatter_step_kernel(frontier_2, frontier, mask, image, Scalar(0))
    c = res.c[]
    frontier_2 = res.frontier_2
    while c > 0
        (frontier_2, frontier) = (frontier, frontier_2)
        res = fill_finch_scatter_step_kernel(frontier_2, frontier, mask, image, Scalar(0))
        c = res.c[]
        frontier_2 = res.frontier_2
    end
    return mask
end

function fill_finch_scatter((img, x, y),)
    (xs, ys) = size(img)

    frontier_2 = Tensor(Dense(SparseByteMap(Pattern())), undef, xs, ys)
    frontier = Tensor(Dense(SparseByteMap(Pattern())), undef, xs, ys)
    mask = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    image = Tensor(Dense(Dense(Element(false))), img)
    time = @belapsed fill_finch_scatter_kernel($frontier_2, $frontier, $mask, $image, $x, $y)
    return (;time=time, mem = summarysize(mask), nnz = countstored(mask), output=mask)
end