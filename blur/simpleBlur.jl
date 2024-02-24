#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

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

eval(Finch.@finch_kernel function blurSimple(input, output, tmp)
    output .= 0
    for y = _
        tmp .= 0
        for x = _
            for c = _
                tmp[c, x] += (coalesce(input[c, ~(x-1), y], 0) + coalesce(input[c, x, y],0) + coalesce(input[c, ~(x+1), y], 0))/3
            end
        end
        for x = _
            for c = _
                output[c, x, y] += (coalesce(tmp[c, ~(x-1)],0) + coalesce(tmp[c, x], 0) + coalesce(tmp[c, ~(x+1)],0))/3
            end
        end
    end
end)

function runBlurSimple(input, output, tmp)
    blurSimple(input, output, tmp)
end

input = Tensor(Dense(Dense(RepeatRLE(Float64(0)))))
output = Tensor(Dense(Dense(RepeatRLE(Float64(0)))))
tmp = Tensor(Dense(RepeatRLE(Float64(0))))

eval(Finch.@finch_kernel function blurRLE(input, output, tmp)
    output .= 0
    for x = _
        tmp .= 0
        for c = _
            for y = _
                tmp[y, c] += (coalesce(input[y, c, ~(x-1)], 0) + coalesce(input[y, c, x],0) + coalesce(input[y, c, ~(x+1)], 0))/3
            end
        end
        for c = _
            for y = _
                output[y, c, x] += (coalesce(tmp[~(y-1), c], 0) + coalesce(tmp[y, c], 0) + coalesce(tmp[~(y+1), c],0))/3
            end
        end
    end
end)

function convertImageToFinch(img)
    (cs, xs, ys) = size(img)
    inp = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img)
    outBuff = zeros(Float64, (cs, xs, ys))
    out = Tensor(Dense(Dense(Dense(Element(Float64(0))))), outBuff)
    tempBuf = zeros(Float64, (xs, cs))
    tmp = Tensor(Dense(Dense(Element(Float64(0)))), tempBuf)
    return (inp, out, tmp)
end

function convertImageToFinchRLE(img)
    img = permutedims(img, 3, 1, 2)
    (ys, cs, xs) = size(img)
    inp = Tensor(Dense(Dense(RepeatRLE(Float64(0)))), img)
    out = Tensor(Dense(Dense(RepeatRLE(Float64(0)))), ys, cs, xs)
    tmp = Tensor(Dense(RepeatRLE(Float64(0))), ys, cs)
    return (inp, out, tmp)
end

function testCorrect(img1, img2)
    img2AsDense = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img2)
    img1AsDense = Tensor(Dense(Dense(Dense(Element(Float64(0))))), img1)
    return img2AsDense == img1AsDense
end

function runBlurRLE(input, output, tmp)
    blurRLE(input, output, tmp)
end


function runOnImage(filename)
    data = testimage(filename)
    data_raw = Array{Float64}((channelview(data)))
    finchData = convertImageToFinch(data_raw)
    finchRLEData = convertImageToFinch(data_raw)
    println(sizeof(finchData[1]))
    println(sizeof(finchRLEData[1]))
    data1 = openCVBlur(data_raw)
    runBlurSimple(finchData[1], finchData[2], finchData[3])

    correct = testCorrect(finchData[2], data1)

    timeFinch = @belapsed runBlurSimple($(finchData[1]), $(finchData[2]), $(finchData[3])) evals=1
    timeFinchRLE = @belapsed runBlurSimple($(finchRLEData[1]), $(finchRLEData[2]), $(finchRLEData[3])) evals=1
    timeOpenCV = @belapsed openCVBlur($data_raw) evals=1

    result = Dict("imagename"=>filename, "finchTime"=>timeFinch, "openCVtime"=>timeOpenCV, "finchRLETime"=>timeFinchRLE, 
        "name"=>"box-blur3x3-Float64",
        "type"=>"Float64", "sizex"=>3, "sizey"=>3,
        "correct" => correct)
    return result
end


function main(resultfile)
    images = [("mandrill.tiff", Float64)]
    for img in images
        ret = runOnImage(img[1]) # extend with float variatio when it works.
        write(resultfile, JSON.json(ret, 4))
    end

end


main("test.json")
