for kernel in Serialization.deserialize(joinpath(@__DIR__, "hist_kernels.jls"))
    eval(kernel)
end

function hist_finch(img)
    return (mask) -> begin
        bins = Tensor(Dense(Element(0)), undef, 16)
        img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
        mask = Tensor(Dense(Dense(Element(false))), mask)
        time = @belapsed hist_finch_kernel($bins, $img, $mask)
        result = hist_finch_kernel(bins, img, mask)
        (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
    end
end

function hist_finch_rle(img)
    return (mask) -> begin
        bins = Tensor(Dense(Element(0)), undef, 16)
        img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
        mask = Tensor(Dense(SparseRLE(Pattern())), mask .!= 0)
        time = @belapsed hist_finch_kernel($bins, $img, $mask)
        result = hist_finch_kernel(bins, img, mask)
        (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
    end
end

function hist_opencv(img)
    return (mask) -> begin
        # step 1: gray scale the image
        # vector of images - don't ask
        imgs = Vector{OpenCV.InputArray}([reshape(img, 1, size(img)...)])
        mask = reshape(mask, 1, size(mask)...)
        result = Ref{Any}()
        channels = Vector{Int32}([0])
        bins = Vector{Int32}([16])
        regions = Vector{Float32}([0, 256])
        time = @belapsed OpenCV.calcHist($imgs, $channels, $mask, $bins, $regions)
        result = OpenCV.calcHist(imgs, channels, mask, bins, regions)
        (;time=time, output=map(x->round(Int, x), reshape(Array(result), :)), mem = summarysize(img), nnz = length(img))
    end
end

function histblur_finch(img)
    return (mask) -> begin
        bins = Tensor(Dense(Element(0)), undef, 16)
        img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
        mask = Tensor(Dense(Dense(Element(false))), mask)
        tmp = Tensor(Dense(Element(0)))
        time = @belapsed histblur_finch_kernel($bins, $img, $tmp, $mask)
        result = histblur_finch_kernel(bins, img, tmp, mask)
        (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
    end
end

function histblur_finch_rle(img)
    return (mask) -> begin
        bins = Tensor(Dense(Element(0)), undef, 16)
        img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
        mask = Tensor(Dense(SparseRLE(Pattern())), mask .!= 0)
        tmp = Tensor(Dense(Element(0)))
        time = @belapsed histblur_finch_kernel($bins, $img, $tmp, $mask)
        result = histblur_finch_kernel(bins, img, tmp, mask)
        (;time=time, output=result.bins, mem = summarysize(img), nnz = countstored(img))
    end
end

function histblur_opencv_kernel(img, channels, mask, bins, regions)
    img = OpenCV.blur(reshape(img, 1, size(img)...), blur_opencv_kernelSize; borderType=OpenCV.BORDER_CONSTANT)
    imgs = Vector{OpenCV.InputArray}([img])
    OpenCV.calcHist(imgs, channels, mask, bins, regions)
end
function histblur_opencv(img)
    return (mask) -> begin
        # step 1: gray scale the image
        # vector of images - don't ask
        mask = reshape(mask, 1, size(mask)...)
        result = Ref{Any}()
        channels = Vector{Int32}([0])
        bins = Vector{Int32}([16])
        regions = Vector{Float32}([0, 256])
        time = @belapsed histblur_opencv_kernel($img, $channels, $mask, $bins, $regions)
        result = histblur_opencv_kernel(img, channels, mask, bins, regions)
        (;time=time, output=map(x->round(Int, x), reshape(Array(result), :)), mem = summarysize(img), nnz = length(img))
    end
end

function blur_finch(image)
    return (mask) -> begin
        output = Tensor(Dense(Dense(Element(UInt8(0)))))
        image = Tensor(Dense(Dense(Element(UInt8(0)))), image)
        tmp = Tensor(Dense(Element(UInt(0))))
        mask = Tensor(Dense(Dense(Element(false))), mask)
        time = @belapsed blur_finch_kernel($output, $image, $tmp, $mask)
        blurry = blur_finch_kernel(output, image, tmp, mask).output
        (;time=time, output=blurry, mem = summarysize(image), nnz = countstored(image))
    end
end

function blur_finch_rle(image)
    return (mask) -> begin
        output = Tensor(Dense(Dense(Element(UInt8(0)))))
        image = Tensor(Dense(Dense(Element(UInt8(0)))), image)
        tmp = Tensor(Dense(Element(UInt(0))))
        mask = Tensor(Dense(SparseRLE(Pattern())), mask .!= 0)
        time = @belapsed blur_finch_kernel($output, $image, $tmp, $mask)
        blurry = blur_finch_kernel(output, image, tmp, mask).output
        (;time=time, output=blurry, mem = summarysize(image), nnz = countstored(image))
    end
end

const blur_opencv_kernelSize = OpenCV.Size(Int32(3), Int32(3))
blur_opencv_kernel(image) = begin
    OpenCV.blur(reshape(image, 1, size(image)...), blur_opencv_kernelSize; borderType=OpenCV.BORDER_CONSTANT)
end

function blur_opencv(image)
    return (mask) -> begin


        time = @belapsed blur_opencv_kernel($image)
        blurry = blur_opencv_kernel(image)
        (;time=time, output=reshape(Array(blurry), size(image)) .* mask, mem = summarysize(image), nnz = length(image))
    end
end

