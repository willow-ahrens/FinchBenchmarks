using Finch
using TestImages
using ImageCore, OpenCV, TestImages, MosaicViews, Colors
using BenchmarkTools
using JSON

function openCVBlur(data)
    img_blur = OpenCV.blur(data, OpenCV.Size(Int32(3), Int32(3)))
    return img_blur
end

input = Tensor(Dense(Dense(Dense(Element(Float64(0))))))
output = Tensor(Dense(Dense(Dense(Element(Float64(0))))))
tmp = Tensor(Dense(Dense(Element(Float64(0)))))


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
                output[c, x, y] += (coalesce(tmp[~(y-1), c],0) + coalesce(tmp[~y, c], 0) + coalesce(tmp[~(y+1), c],0))/3
            end
        end
    end
end



@inline function runBlurSimple(input, output, tmp)
    blurSimple(input, output, tmp)
end


function convertImageToFinch(img)
    (cs, xs, ys) = size(img)
    inp = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img)
    outBuff = zeros(Float64, (cs, xs, ys))
    out = Tensor(Dense(Dense(Dense(Element(Float64(0))))), outBuff)
    tempBuf = zeros(Float64, (xs, cs))
    tmp = Tensor(Dense(Dense(Element(Float64(0)))), tempBuf)
    return (inp, out, tmp)
end

function testCorrect(img1, img2)
    img2AsDense = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img2)
    img1AsDense = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img1)
    return img2AsDense == img1AsDense
end


function runOnImage(filename)
    data = testimage(filename)
    data_raw = Array{Float64}((channelview(data)))
    finchData = convertImageToFinch(data_raw)
    data1 = openCVBlur(data_raw)
    runBlurSimple(finchData[1], finchData[2], finchData[3])

    correct = testCorrect(finchData[2], data1)


    timeFinch = @belapsed runBlurSimple($(finchData[1]), $(finchData[2]), $(finchData[3])) evals=1
    timeOpenCV = @belapsed openCVBlur($data_raw) evals=1


    result = Dict("imagename"=>filename, "finchTime"=>timeFinch, "openCVtime"=>timeOpenCV,
        "name"=>"box-blur3x3-Float64",
        "type"=>"Float64", "sizex"=>3, "sizey"=>3,
        "correct" => correct)
    return result
end


function main(resultfile)
    images = [("Mandrill", Float64)]
    for img in images
        ret = runOnImage(img[1]) # extend with float variatio when it works.
        open(resultfile,"w") do f
            JSON.print(f, JSON.json(ret))
        end

    end

end


main("test.json")
