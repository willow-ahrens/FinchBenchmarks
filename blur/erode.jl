for kernel in Serialization.deserialize(joinpath(@__DIR__, "erode_kernels.jls"))
    eval(kernel)
end

erode_opencv_kernel(data, filter, niters) = OpenCV.erode(data, filter, iterations=niters)

function erode_opencv((img, niters),)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed erode_opencv_kernel($input, $filter, $niters) evals=1
    output = dropdims(Array(erode_opencv_kernel(input, filter, niters)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

erode_finch_kernel2(output, input, tmp, niters) = begin
    (output, input) = (input, output)
    for i in 1:niters
        (output, input) = (input, output)
        output = erode_finch_kernel(output, input, tmp).output
    end
    return output
end

function erode_finch((img, niters),)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed erode_finch_kernel2($output, $input, $tmp, $niters) evals=1
    input = Tensor(Dense(Dense(Element(false))), img)
    output = erode_finch_kernel2(output, input, tmp, niters)
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

erode_finch_bits_kernel2(output, input, tmp, niters) = begin
    (output, input) = (input, output)
    for i in 1:niters
        (output, input) = (input, output)
        output = erode_finch_bits_kernel(output, input, tmp).output
    end
    return output
end

function erode_finch_bits((img, niters),)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_kernel2($outputb, $inputb, $tmpb, $niters) evals=1
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = erode_finch_bits_kernel2(outputb, inputb, tmpb, niters)
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

function erode_finch_rle((img, niters),)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseRLE(Pattern())), undef, xs, ys)
    tmp = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    time = @belapsed erode_finch_kernel2($output, $input, $tmp, $niters) evals=1
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = erode_finch_kernel2(output, input, tmp, niters)
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

erode_finch_bits_mask_kernel2(output, input, tmp, mask, niters) = begin
    (output, input) = (input, output)
    for i in 1:niters
        (output, input) = (input, output)
        output = erode_finch_bits_mask_kernel(output, input, tmp,  mask).output
    end
    return output
end

function erode_finch_bits_mask((img, niters),)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    maskb = Tensor(Dense(SparseList(Pattern())), imgb .!= 0)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Element(UInt(0))), undef, xb)
    time = @belapsed erode_finch_bits_mask_kernel2($outputb, $inputb, $tmpb, $maskb, $niters) evals=1
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = erode_finch_bits_mask_kernel2(outputb, inputb, tmpb, maskb, niters)
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end