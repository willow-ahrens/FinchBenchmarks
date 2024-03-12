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
        data .= 0
        data[x, y] = 1
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