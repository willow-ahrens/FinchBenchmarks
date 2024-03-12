for kernel in Serialization.deserialize(joinpath(@__DIR__, "fill_kernels.jls"))
    eval(kernel)
end

function fill_opencv_kernel(mask, filter, x, y)
    data = zeros(UInt8, size(mask)...)
    data[1, x, y] = UInt8(1)
    data2 = OpenCV.multiply(OpenCV.dilate(data, filter), mask)
    while data_2 != data
        data = data_2
        data2 = OpenCV.multiply(OpenCV.dilate(data, filter), mask)
    end
    return data2
end

function fill_opencv(img, (x, y))
    mask = Array{UInt}(reshape(img .!= 0, 1, size(img)...))
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed fill_opencv_kernel($data, $mask, filter) evals=1
    output = dropdims(Array(dilate_opencv_kernel(data, mask, filter)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

function fill_finch_kernel2(data2, data, mask, tmp, x, y)
    @finch begin
        data .= 0
        data[x, y] = 1
    end
    fill_finch_kernel(data2, data, mask, tmp)
    while data2 != data
        (data, data2) = (data2, data)
        fill_finch_kernel(data2, data, mask, tmp)
    end
end

function fill_finch(img, (x, y))
    (xs, ys) = size(img)
    mask = Tensor(Dense(Dense(Element(false))), img)
    data = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    data2 = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed fill_finch_kernel2($data2, $data, $mask, $tmp, x, y) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end