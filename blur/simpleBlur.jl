using Finch
using MatrixDepot
using TestImages
using ImageCore, OpenCV, TestImages, MosaicViews
using BenchmarkTools


# img_orig = testimage("Mandrill")
# img_raw =  collect(rawview(channelview(img_orig)))
# img_gray = OpenCV.cvtColor(img_raw, OpenCV.COLOR_RGB2GRAY)


function openCVBlur(data)
    img_blur = OpenCV.blur(data, OpenCV.Size(Int32(3), Int32(3)))
    return img_blur
end

input = Fiber!(Dense(Dense(Dense(Element(UInt8(0))))))
output = Fiber!(Dense(Dense(Dense(Element(UInt8(0))))))
tmp = Fiber!(Dense(Dense(Element(UInt8(0)))))


@Finch.finch_kernel function blurSimple(input, output, tmp)
    output .= 0
    for x = _
        tmp .= 0
        for y = _
            for c = _
                tmp[y, c] += (coalesce(input[c, y, ~(x-1)], 0) + coalesce(input[c, y, ~x],0) + coalesce(input[c, y, ~(x+1)], 0))/3
            end
        end
        for y = _
            for c = _
                output[c, x, y] += (coalesce(tmp[~(y-1), c],0) + coalesce(tmp[y, c], 0) + coalesce(tmp[~(y+1), c],0))/3
            end
        end
    end
end



function convertImageToFinch(img)
    (cs, xs, ys) = size(img)
    inp = Fiber!(Dense(Dense(Dense(Element(UInt8(0))))), img)
    outBuff = zeros(UInt8, (cs, xs, ys))
    out = Fiber!(Dense(Dense(Dense(Element(UInt8(0))))), outBuff)
    tempBuf = zeros(UInt8, (xs, cs))
    tmp = Fiber!(Dense(Dense(Element(UInt8(0)))), tempBuf)
    return (inp, out, tmp)
end


function runOnImage(filename)
    data = testimage(filename)
    data_raw = collect(rawview(channelview(data)))
    finchData = convertImageToFinch(data_raw)
    data1 = openCVBlur(data_raw)
    blurSimple(finchData[1], finchData[2], finchData[3])
    #compare both
    # unit 8 issue
    #benchmark
end
