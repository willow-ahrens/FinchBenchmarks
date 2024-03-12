for kernel in Serialization.deserialize(joinpath(@__DIR__, "fill_kernels.jls"))
    eval(kernel)
end

function fill_opencv_kernel(mask, filter, x, y)
    data = zeros(UInt8, size(mask)...)
    data[1, x, y] = 0x01
    data_2 = OpenCV.multiply(OpenCV.dilate(data, filter), mask)
    while data_2 != data
        data = data_2
        data_2 = OpenCV.multiply(OpenCV.dilate(data, filter), mask)
    end
    return data_2
end

function fill_opencv((img, x, y),)
    mask = Array{UInt8}(reshape(img .!= 0, 1, size(img)...))
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed fill_opencv_kernel($mask, $filter, $x, $y) evals=1
    output = dropdims(Array(fill_opencv_kernel(mask, filter, x, y)), dims=1)
    return (; time = time, mem = summarysize(img), nnz = length(img), output = output)
end

function fill_finch_kernel2(data2, data, mask, tmp, tmp2, x, y)
    @finch begin
        data .= false
        data[x, y] = true
    end
    fill_finch_kernel(data2, data, mask, tmp, tmp2)
    while data2 != data
        (data, data2) = (data2, data)
        fill_finch_kernel(data2, data, mask, tmp, tmp2)
    end
    return data2
end

function fill_finch((img, x, y),)
    (xs, ys) = size(img)
    mask = Tensor(Dense(Dense(Element(false))), img)
    data = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    data2 = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    tmp2 = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed fill_finch_kernel2($data2, $data, $mask, $tmp, $tmp2, $x, $y) evals=1
    output = fill_finch_kernel2(data2, data, mask, tmp, tmp2, x, y)
    return (;time=time, mem = summarysize(output), nnz = countstored(output), output=output)
end

#=
function fill_finch_bits_kernel2(data2, data, mask, tmp, tmp2, x, y)
    @finch begin
        data .= false
        data[x, y] = true
    end
    fill_finch_bits_kernel(data2, data, mask, tmp, tmp2)
    while data2 != data
        (data, data2) = (data2, data)
        fill_finch_bits_kernel(data2, data, mask, tmp, tmp2)
    end
    return data2
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
function fill_finch_bits((img, x, y),)
    (xs, ys) = size(img)
    imgb = pack_bits(img .!= 0x00)
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    mask = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    data = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    data2 = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmp = Tensor(Dense(Element(UInt(0))), undef, xb)
    tmp2 = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed fill_finch_kernel2($data2, $data, $mask, $tmp, $tmp2, $x, $y) evals=1
    output = fill_finch_kernel2(data2, data, mask, tmp, tmp2, x, y)
    return (;time=time, mem = summarysize(output), nnz = countstored(output), output=output)
end
=#

function fill_finch_rle((img, x, y),)
    (xs, ys) = size(img)
    mask = Tensor(Dense(SparseRLE(Pattern())), img .!= 0x00)
    data = Tensor(Dense(SparseRLE(Pattern())), undef, xs, ys)
    data2 = Tensor(Dense(SparseRLE(Pattern())), undef, xs, ys)
    tmp = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    tmp2 = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    time = @belapsed fill_finch_kernel2($data2, $data, $mask, $tmp, $tmp2, $x, $y) evals=1
    output = fill_finch_kernel2(data2, data, mask, tmp, tmp2, x, y)
    return (;time=time, mem = summarysize(output), nnz = countstored(output), output=output)
end

function fill_finch_rle2((img, x, y),)
    (xs, ys) = size(img)
    data2 = Tensor(SparseList(SparseRLE(Pattern())), undef, xs, ys)
    data = Tensor(SparseList(SparseRLE(Pattern())), undef, xs, ys)
    mask = Tensor(Dense(SparseRLE(Pattern())), img .!= 0x00)
    tmp = Tensor(SparseList(SparseRLE(Pattern(), merge=false)), undef, xs, ys)
    tmp2 = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    time = @belapsed fill_finch_kernel2($data2, $data, $mask, $tmp, $tmp2, $x, $y) evals=1
    output = fill_finch_kernel2(data2, data, mask, tmp, tmp2, x, y)
    return (;time=time, mem = summarysize(output), nnz = countstored(output), output=output)
end