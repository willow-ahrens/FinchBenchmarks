img = Tensor(Dense(Dense(Element(UInt8(0)))))
mask = Tensor(Dense(Dense(Element(UInt8(0)))))
bins = Tensor(Dense(Element(false)))

for kernel in Serialization.deserialize(joinpath(@__DIR__, "hist_kernels.jls"))
    eval(kernel)
end

function hist_finch(img)
    return (mask) -> begin
        img = Tensor(Dense(Dense(Element(UInt8(0)))), img)
        mask = Tensor(Dense(Dense(Element(UInt8(0)))), img)
        bins = Tensor(Dense(Element(false)), undef, 16)
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
        regions = Vector{Float32}([0, 255])
        time = @belapsed OpenCV.calcHist($imgs, $channels, $mask, $bins, $regions)
        result = OpenCV.calcHist(imgs, channels, mask, bins, regions)
        (;time=time, output=reshape(Array(result), :), mem = summarysize(img), nnz = length(img))
    end
end