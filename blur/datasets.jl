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

function willow_gen(n)
    () -> [UInt8((i - n/2)^2 + (j - n/2)^2 < (n/4)^2) for i in 1:n, j in 1:n, k = 1:1]
end