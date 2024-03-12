for kernel in Serialization.deserialize(joinpath(@__DIR__, "erode_kernels.jls"))
    eval(kernel)
end

erode_opencv_kernel(data, filter) = OpenCV.erode(data, filter)

function erode_opencv(img)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed erode_opencv_kernel($input, $filter) evals=1
    output = dropdims(Array(erode_opencv_kernel(input, filter)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

dilate_opencv_kernel(data, filter) = OpenCV.dilate(data, filter)

function dilate_opencv(img)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed dilate_opencv_kernel($input, $filter) evals=1
    output = dropdims(Array(dilate_opencv_kernel(input, filter)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

function erode_finch(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function dilate_finch(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed dilate_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function erode_finch_bits(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function dilate_finch_bits(img)
    (xs, ys) = size(img)
    imgb = pack_bits(img .!= 0x00)
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed dilate_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_bits_sparse(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(SparseList(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(SparseList(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(SparseList(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_bits_rle(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(DenseRLE(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(DenseRLE(Element(UInt(0)), merge=false)), undef, xb, ys)
    tmpb = Tensor(DenseRLE(Element(UInt(0)), merge=false), undef, xb)
    time = @belapsed erode_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    outputb = erode_finch_bits_kernel(outputb, inputb, tmpb).output
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_bits_mask(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    maskb = Tensor(Dense(SparseList(Pattern())), imgb .!= 0)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_mask_kernel($outputb, $inputb, $tmpb, $maskb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_rle(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseRLE(Pattern(), merge=false)), undef, xs, ys)
    tmp = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function erode_finch_sparse(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseList(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseList(Pattern())), undef, xs, ys)
    tmp = Tensor(SparseList(Pattern()), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end
