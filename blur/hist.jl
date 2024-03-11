function finch_hist_serial_kernel(result, img, mask)
    @finch begin
        result .= 0 
        for x=_
            for y=_
                if mask[1, y, x] > UInt8(0)
                    result[div(img[1, y, x], 16) + 1] += 1
                end
            end
        end
    end
    return result
end


function finch_hist_run(f, img, nthr)
    #step -1: set num Threads
    OpenCV.setNumThreads(nthr)
    # step 0: setup results
    results = Vector{Float32}([0 for _ in 1:16])
    # step 1: gray scale the image
    # vector of images - don't ask
    imgs = Vector{OpenCV.InputArray}([])
    img_raw =  collect(rawview(channelview(img)))
    img_gray = OpenCV.cvtColor(img_raw, OpenCV.COLOR_RGB2GRAY)
    push!(imgs, img_gray)
    mask = ones(UInt8, size(img_gray))
    result = Ref{Any}()
    channels = Vector{Int32}([0])
    bins = Vector{Int32}([16])
    regions = Vector{Float32}([0.0, 256.0])
    time = @belapsed $result[] = $f($results, $img_gray, $mask)
    (;time=time, result=result[])
end

finch_hist_serial(img, nthr) = finch_hist_run(finch_hist_serial_kernel, img, nthr)

function opencv_hist(img, nthr)
    #step -1: set num Threads
    OpenCV.setNumThreads(nthr)
    # step 0: setup results
    results = Vector{Float32}([0 for _ in 1:16])
    # step 1: gray scale the image
    # vector of images - don't ask
    imgs = Vector{OpenCV.InputArray}([])
    img_raw =  collect(rawview(channelview(img)))
    img_gray = OpenCV.cvtColor(img_raw, OpenCV.COLOR_RGB2GRAY)
    push!(imgs, img_gray)
    mask = ones(UInt8, size(img_gray))
    result = Ref{Any}()
    channels = Vector{Int32}([0])
    bins = Vector{Int32}([16])
    regions = Vector{Float32}([0.0, 256.0])
    time = @belapsed $result[] = OpenCV.calcHist($imgs, $channels, OpenCV.Mat($mask), $bins, $regions)
    (;time=time, result=reshape(Array(result[]), 16))

end