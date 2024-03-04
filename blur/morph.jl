#!/usr/bin/env julia
if abspath(PROGRAM_FILE) == @__FILE__
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using MLDatasets
using Finch
#using TestImages
using OpenCV#, TestImages, MosaicViews, Colors, Images, FileIO
using BenchmarkTools
using LinearAlgebra
using JSON
using Base: summarysize

download_cache = joinpath(@__DIR__, "../cache")

using ZipFile
using Images
using MLDatasets
using Downloads
using DelimitedFiles
using GZip
using CSV
using DataFrames

function unzip(file,exdir="",flatten=false)
    fileFullPath = isabspath(file) ?  file : joinpath(pwd(),file)
    basePath = dirname(fileFullPath)
    outPath = (exdir == "" ? basePath : (isabspath(exdir) ? exdir : joinpath(pwd(),exdir)))
    isdir(outPath) ? "" : mkdir(outPath)
    zarchive = ZipFile.Reader(fileFullPath)
    for f in zarchive.files
        fullFilePath = joinpath(outPath,f.name)
        if flatten 
            if !(endswith(f.name,"/") || endswith(f.name,"\\"))
                write(joinpath(outPath,basename(fullFilePath)), read(f))
            end
        else
            if (endswith(f.name,"/") || endswith(f.name,"\\"))
                mkdir(fullFilePath)
            else
                write(fullFilePath, read(f))
            end
        end
    end
    close(zarchive)
end

function download_dataset(url, name)
    path = joinpath(download_cache, name)
    fname = joinpath(path, basename(url))
    if !isfile(fname)
        mkpath(path)
        Downloads.download(url, fname)
        return fname, true
    else
        return fname, false
    end
end

function download_humansketches()
    sketches_link = "https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip"
    loc, unpack = download_dataset(sketches_link, "sketches")
    unzip_path = joinpath(dirname(loc), "pngs")
    if unpack
        unzip(loc, unzip_path, true)
    end
    return unzip_path
end

"""
humansketches dataset tensor
========================
humansketches([idxs])

Return a 3-tensor A[sketch number, vertical pixel position, horizontal pixel
position], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. `idxs` is an optional list specifying which sketch images to
load. The sketches number from 1:20_000.
"""
function humansketches(idxs = 1:100)
    @boundscheck begin
        extrema(idxs) ⊆ 1:20_000 || throw(BoundsError("humansketches", idxs))
    end

    path = download_humansketches()

    out = Array{Gray{N0f8}, 3}(undef, 1111,1111, length(idxs))

    for (n, i) in enumerate(idxs)
        out[:, :, n] = load(joinpath(path, "$i.png"))
    end
    return out
end

"""
mnist_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from mnist.
"""
function mnist_train()
    dir = joinpath(download_cache, "mnist")
    MNIST(:train, dir=dir, Tx=UInt8).features
end

"""
fashionmnist_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from fashionmnist.
"""
function fashionmnist_train()
    dir = joinpath(download_cache, "fashionmnist")
    FashionMNIST(:train, dir=dir, Tx=UInt8).features
end

"""
omniglot_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from omniglot.
"""
function omniglot_train()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:train, dir=dir, Tx=UInt8).features
end

"""
mnist_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from mnist.
"""
function mnist_test()
    dir = joinpath(download_cache, "mnist")
    MNIST(:test, dir=dir, Tx=UInt8).features
end

"""
fashionmnist_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from fashionmnist.
"""
function fashionmnist_test()
    dir = joinpath(download_cache, "fashionmnist")
    FashionMNIST(:test, dir=dir, Tx=UInt8).features
end

"""
omniglot_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from omniglot.
"""
function omniglot_test()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:test, dir=dir, Tx=UInt8).features
end

"""
omniglot_small1 dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the small1 split from omniglot.
"""
function omniglot_small1()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:small1, dir=dir, Tx=UInt8).features
end

"""
omniglot_small2 dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the small2 split from omniglot.
"""
function omniglot_small2()
    dir = joinpath(download_cache, "omniglot")
    Omniglot(:small2, dir=dir, Tx=UInt8).features
end

"""
emnist_digits_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from the digits extension of emnist.
"""
function emnist_digits_test()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:digits, :test, dir=dir, Tx=UInt8).features
end

"""
emnist_digits_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from the digits extentsion of mnist.
"""
function emnist_digits_train()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:digits, :train, dir=dir, Tx=UInt8).features
end

"""
emnist_letters_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from the letters extension of emnist.
"""
function emnist_letters_test()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:letters, dir=dir, Tx=UInt8).features
end

"""
emnist_letters_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from the letters extentsion of mnist.
"""
function emnist_letters_train()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:letters, :train, dir=dir, Tx=UInt8).features
end

"""
emnist_test dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the test split from the complete emnist.
"""
function emnist_test()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:byclass, :test, dir=dir, Tx=UInt8).features
end

"""
emnist_train dataset tensor
========================
Return a 3-tensor A[vertical pixel position, horizontal pixel position, image
number], measured from image upper left. Pixel values are stored using 8-bit
grayscale values. This returns the train split from complete emnist.
"""
function emnist_train()
    dir = joinpath(download_cache, "emnist")
    EMNIST(:byclass, :train, dir=dir, Tx=UInt8).features
end

erode_opencv_kernel(data, filter) = OpenCV.erode(data, filter)

function erode_opencv(img)
    input = reshape(img, 1, size(img)...)
    filter = ones(Int8, 1, 3, 3)
    time = @belapsed erode_opencv_kernel($input, $filter) evals=1
    output = dropdims(Array(erode_opencv_kernel(input, filter)), dims=1)
    return (; time = time, mem = summarysize(input), nnz = length(input), output = output)
end

input = Tensor(Dense(Dense(Element(false))))
output = Tensor(Dense(Dense(Element(false))))
tmp = Tensor(Dense(Element(false)))

eval(Finch.@finch_kernel function erode_finch_kernel(output, input, tmp)
    output .= false
    for y = _
        tmp .= false
        for x = _
            tmp[x] = coalesce(input[x, ~(y-1)], true) & input[x, y] & coalesce(input[x, ~(y+1)], true)
        end
        for x = _
            output[x, y] = coalesce(tmp[~(x-1)], true) & tmp[x] & coalesce(tmp[~(x+1)], true)
        end
    end
end)

function erode_finch(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(Dense(Element(false))), img)
    output = Tensor(Dense(Dense(Element(false))), undef, xs, ys)
    tmp = Tensor(Dense(Element(false)), undef, xs)
    time = @belapsed erode_finch_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

input = Tensor(Dense(Dense(Element(UInt(0)))))
output = Tensor(Dense(Dense(Element(UInt(0)))))
tmp = Tensor(Dense(Dense(Element(UInt(0)))))

eval(Finch.@finch_kernel function erode_finch_bits_kernel(output, input, tmp)
    tmp .= 0
    for y = _
        for x = _
            tmp[x, y] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
        end
    end
    output .= 0
    for y = _
        for x = _
            let tl = coalesce(tmp[~(x-1), y], ~(UInt(0))), t = tmp[x, y], tr = coalesce(tmp[~(x+1), y], ~(UInt(0)))
                output[x, y] = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
            end
        end
    end
end)

function pack_bits(img)
    xs, ys = size(img)
    xb = cld(xs + 1, 64)
    imgb = fill(UInt(0), xb, ys)
    for y in 1:ys
        for x in 1:xs
            imgb[fld1(x, 64), y] |= UInt(Bool(img[x, y])) << (mod1(x, 64) - 1)
        end
    end
    imgb
end

function unpack_bits(imgb, xs, ys)
    img = zeros(UInt8, xs, ys)
    for y in 1:ys
        for x in 1:xs
            img[x, y] = UInt8((imgb[fld1(x, 64), y] >> (mod1(x, 64) - 1)) & 0x01)
        end
    end
    img
end

function erode_finch_bits(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    time = @belapsed erode_finch_bits_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

input = Tensor(Dense(SparseList(Element(UInt(0)))))
output = Tensor(Dense(SparseList(Element(UInt(0)))))
tmp = Tensor(Dense(SparseList(Element(UInt(0)))))

eval(Finch.@finch_kernel function erode_finch_bits_sparse_kernel(output, input, tmp)
    tmp .= 0
    for y = _
        for x = _
            tmp[x, y] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
        end
    end
    output .= 0
    for y = _
        for x = _
            let tl = coalesce(tmp[~(x-1), y], ~(UInt(0))), t = tmp[x, y], tr = coalesce(tmp[~(x+1), y], ~(UInt(0)))
                output[x, y] = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
            end
        end
    end
end)

function erode_finch_bits_sparse(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(SparseList(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(SparseList(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(SparseList(Element(UInt(0)))), undef, xb, ys)
    time = @belapsed erode_finch_bits_sparse_kernel($outputb, $inputb, $tmpb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

input = Tensor(Dense(DenseRLE(Element(UInt(0)))))
output = Tensor(Dense(DenseRLE(Element(UInt(0)), merge=false)))
tmp = Tensor(Dense(DenseRLE(Element(UInt(0)), merge=false)))

eval(Finch.@finch_kernel function erode_finch_bits_rle_kernel(output, input, tmp)
    tmp .= 0
    for y = _
        for x = _
            tmp[x, y] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
        end
    end
    output .= 0
    for y = _
        for x = _
            let tl = coalesce(tmp[~(x-1), y], ~(UInt(0))), t = tmp[x, y], tr = coalesce(tmp[~(x+1), y], ~(UInt(0)))
                output[x, y] = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
            end
        end
    end
    return output
end)

function erode_finch_bits_rle(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(DenseRLE(Element(UInt(0)))), imgb)
    outputb = Tensor(Dense(DenseRLE(Element(UInt(0)), merge=false)), undef, xb, ys)
    tmpb = Tensor(Dense(DenseRLE(Element(UInt(0)), merge=false)), undef, xb, ys)
    time = @belapsed erode_finch_bits_rle_kernel($outputb, $inputb, $tmpb) evals=1
    outputb = erode_finch_bits_rle_kernel(outputb, inputb, tmpb).output
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end

input = Tensor(Dense(Dense(Element(UInt(0)))))
mask = Tensor(Dense(SparseList(Pattern())))
output = Tensor(Dense(Dense(Element(UInt(0)))))
tmp = Tensor(Dense(Dense(Element(UInt(0)))))

eval(Finch.@finch_kernel function erode_finch_bits_mask_kernel(output, input, tmp, mask)
    tmp .= 0
    for y = _
        for x = _
            if mask[x, y]
                tmp[x, y] = coalesce(input[x, ~(y-1)], ~(UInt(0))) & input[x, y] & coalesce(input[x, ~(y+1)], ~(UInt(0)))
            end
        end
    end
    output .= 0
    for y = _
        for x = _
            if mask[x, y]
                let tl = coalesce(tmp[~(x-1), y], ~(UInt(0))), t = tmp[x, y], tr = coalesce(tmp[~(x+1), y], ~(UInt(0)))
                    output[x, y] = ((tr << (8 * sizeof(UInt) - 1)) | (t >> 1)) & t & ((t << 1) | (tl >> (8 * sizeof(UInt) - 1)))
                end
            end
        end
    end
end)

function erode_finch_bits_mask(img)
    (xs, ys) = size(img)
    imgb = .~(pack_bits(img .== 0x00))
    @assert img == unpack_bits(imgb, xs, ys)
    (xb, ys) = size(imgb)
    inputb = Tensor(Dense(Dense(Element(UInt(0)))), imgb)
    maskb = Tensor(Dense(SparseList(Pattern())), imgb .!= 0)
    outputb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    tmpb = Tensor(Dense(Dense(Element(UInt(0)))), undef, xb, ys)
    time = @belapsed erode_finch_bits_mask_kernel($outputb, $inputb, $tmpb, $maskb) evals=1
    output = unpack_bits(outputb, xs, ys)
    return (;time=time, mem = summarysize(inputb), nnz = countstored(inputb), output=output)
end


input = Tensor(Dense(SparseRLE(Pattern())))
output = Tensor(Dense(SparseRLE(Pattern(), merge=false)))
tmp = Tensor(SparseRLE(Pattern(), merge=false))

eval(Finch.@finch_kernel function erode_finch_rle_kernel(output, input, tmp)
    output .= false
    for y = _
        tmp .= false
        for x = _
            tmp[x] = coalesce(input[x, ~(y-1)], true) & input[x, y] & coalesce(input[x, ~(y+1)], true)
        end
        for x = _
            output[x, y] = coalesce(tmp[~(x-1)], true) & tmp[x] & coalesce(tmp[~(x+1)], true)
        end
    end
end)


function erode_finch_rle(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseRLE(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseRLE(Pattern(), merge=false)), undef, xs, ys)
    tmp = Tensor(SparseRLE(Pattern(), merge=false), undef, xs)
    time = @belapsed erode_finch_rle_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

input = Tensor(Dense(SparseList(Pattern())))
output = Tensor(Dense(SparseList(Pattern())))
tmp = Tensor(SparseList(Pattern()))

eval(Finch.@finch_kernel function erode_finch_sparse_kernel(output, input, tmp)
    output .= false
    for y = _
        tmp .= false
        for x = _
            tmp[x] = coalesce(input[x, ~(y-1)], true) & input[x, y] & coalesce(input[x, ~(y+1)], true)
        end
        for x = _
            output[x, y] = coalesce(tmp[~(x-1)], true) & tmp[x] & coalesce(tmp[~(x+1)], true)
        end
    end
end)

function erode_finch_sparse(img)
    (xs, ys) = size(img)
    input = Tensor(Dense(SparseList(Pattern())), Array{Bool}(img))
    output = Tensor(Dense(SparseList(Pattern())), undef, xs, ys)
    tmp = Tensor(SparseList(Pattern()), undef, xs)
    time = @belapsed erode_finch_sparse_kernel($output, $input, $tmp) evals=1
    return (;time=time, mem = summarysize(input), nnz = countstored(input), output=output)
end

function willow_gen(n)
    () -> [UInt8((i - n/2)^2 + (j - n/2)^2 < (n/4)^2) for i in 1:n, j in 1:n, k = 1:1]
end

function main(resultfile)
    results = []

    for (dataset, getdata, I, f) in [
        ("mnist", mnist_train, 1:1, (img) -> Array{UInt8}(img .> 0x02))
        ("willow100", willow_gen(100), 1:1, identity)
        ("willow200", willow_gen(200), 1:1, identity)
        ("willow400", willow_gen(400), 1:1, identity)
        ("willow800", willow_gen(800), 1:1, identity)
        ("willow1600", willow_gen(1600), 1:1, identity)
        ("willow3200", willow_gen(3200), 1:1, identity)
        ("willow6400", willow_gen(6400), 1:1, identity)
        ("omniglot", omniglot_train, 1:10, (img) -> Array{UInt8}(img .!= 0x00))
        ("humansketches", humansketches, 1:10, (img) -> Array{UInt8}(reinterpret(UInt8, img) .< 0xF0))
    ]
        data = getdata()
        for i in I
            input = f(data[:, :, i])

            reference = nothing

            for kernel in [
                (method = "opencv", fn = erode_opencv),
                (method = "finch", fn = erode_finch),
                (method = "finch_rle", fn = erode_finch_rle),
                (method = "finch_sparse", fn = erode_finch_sparse),
                (method = "finch_bits", fn = erode_finch_bits),
                (method = "finch_bits_sparse", fn = erode_finch_bits_sparse),
                (method = "finch_bits_mask", fn = erode_finch_bits_mask),
                (method = "finch_bits_rle", fn = erode_finch_bits_rle),
            ]

                result = kernel.fn(input)

                reference = something(reference, result.output)
                if reference != result.output
                    display(Array{Bool}(reference))
                    display(Array{Bool}(result.output))
                end
                @assert reference == result.output

                println("$dataset [$i]: $(kernel.method) time: ", result.time, "\tmem: ", result.mem, "\tnnz: ", result.nnz)

                push!(results, Dict("imagename"=>"$dataset[$i]", "method"=> kernel.method, "mem" => result.mem, "nnz" => result.nnz, "time"=>result.time))
                write(resultfile, JSON.json(results, 4))
            end
        end
    end

    return results
end

main("test.json")
