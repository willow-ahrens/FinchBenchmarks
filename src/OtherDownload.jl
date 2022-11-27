# hwd+, ilsvrc, sketches

# sketches: https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/sketches_png.zip
# hwd+ (drive folder): https://drive.google.com/drive/folders/1f2o1kjXLvcxRgtmMMuDkA2PQ5Zato4Or 
# hwd+ (500x500 npy): https://drive.google.com/u/0/uc?id=1CInd1YOC0lsEq4_q089SVb7PkhxtrwyE&export=download
# ilsvrc (kaggle login needed, and its huge!): https://www.kaggle.com/competitions/imagenet-object-localization-challenge/

# using HTTP
using ZipFile
using Images
using MLDatasets
using Downloads
using DelimitedFiles
using GZip
using CSV, DataFrames

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
function humansketches(idxs = 1:20_000)
    @boundscheck begin
        extrema(idxs) ⊆ 1:20_000 || throw(BoundsError("humansketches", idxs))
    end

    path = download_humansketches()

    out = Array{Gray{N0f8}, 3}(undef, length(idxs), 1111,1111)

    for (n, i) in enumerate(idxs)
        out[n, :, :] = load(joinpath(path, "$i.png"))
    end
    return out
end

"""
cifar10_test dataset tensor
========================
Return a 4-tensor A[vertical pixel position, horizontal pixel position, channel,
image number], measured from image upper left. Pixel values are stored using
8-bit grayscale values. This returns the test split from cifar10.
"""
function cifar10_test()
    dir = joinpath(download_cache, "cifar10")
    CIFAR10(:test, dir=dir, Tx=UInt8).features
end

"""
cifar10_train dataset tensor
========================
Return a 4-tensor A[vertical pixel position, horizontal pixel position, channel,
image number], measured from image upper left. Pixel values are stored using
8-bit grayscale values. This returns the train split from cifar10.
"""
function cifar10_train()
    dir = joinpath(download_cache, "cifar10")
    CIFAR10(:train, dir=dir, Tx=UInt8).features
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

"""
census dataset matrix
========================
census()
Return a 2458285×69 matrix of Int32 values from the 1990 US census dataset:
https://archive-beta.ics.uci.edu/ml/datasets/us+census+data+1990.
"""
function census()
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
    loc, unpack = download_dataset(link, "census")
    data = readdlm(loc, ',', Int32; skipstart=1)
    return data
end

"""
covtype dataset matrix
========================
covtype()
Return a 581011×55 matrix of Int16 values from the covertype dataset:
https://archive-beta.ics.uci.edu/ml/datasets/covertype.
"""
function covtype()
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    loc, unpack = download_dataset(link, "covtype")
    data = GZip.open(loc, "r") do io
        readdlm(io, ',', Int16)
    end
end

"""
kddcup dataset matrix
========================
kddcup()
Return a 4898431×42 matrix of Float32 values from the kdd cup 1999 dataset:
https://archive-beta.ics.uci.edu/ml/datasets/kdd+cup+1999+data.
"""
function kddcup()
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/kddcup99-mld/kddcup.data.gz"
    loc, unpack = download_dataset(link, "kddcup")
    path = joinpath(download_cache, "kddcup")
    fname = joinpath(path, "kddcup_processed.csv")

    if unpack
        data = CSV.read(loc, DataFrame; header=false)
    
        for j in [2,3,4,42]
            col = data[:, j]
            unique_elems = unique(col)
            map_el = Dict{String, Int}
            for (i,elem) in enumerate(unique_elems)
                map_el = merge!(map_el,Dict(elem=>i))
            end
    
            f = x -> map_el[x]
            transform!(data, j => (x -> f.(x)))
        end
        select!(data, Not(:42))
        select!(data, Not(:4))
        select!(data, Not(:3))
        select!(data, Not(:2))

        data = Matrix(Float32.(data))
        writedlm(fname, data, ',')    
        return data
    end

    return Matrix(CSV.read(fname, DataFrame; types=Float32, header=false))
end

"""
poker dataset matrix
========================
poker()
Return a 25010×11 matrix of Int32 values from a dataset of poker hands:
https://archive-beta.ics.uci.edu/ml/datasets/poker+hand.
"""
function poker()
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data"
    loc, unpack = download_dataset(link, "poker")
    data = readdlm(loc, ',', Int8)
    return data
end

"""
power dataset matrix
========================
power()
Return a 2049280×13 matrix of Float32 values from a dataset of power consumption:
https://archive-beta.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption.
"""
function power()
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    loc, unpack = download_dataset(link, "power")
    unzip_path = dirname(loc)
    if unpack
        unzip(loc, unzip_path, true)
    end

    fname = joinpath(unzip_path, "household_power_consumption.txt")

    data = CSV.read(fname, DataFrame; missingstring=["?", ""], types=Dict(1 => String, 2=> String))

    data = subset(data, All() .=> ByRow(!ismissing))

    tmp1 = split.(data.Date, "/")
    insertcols!(data, [n => parse.(Float32, getindex.(tmp1, i)) for (i, n) in enumerate([:day, :month, :year])]...)
    select!(data, Not(:Date))
    tmp2 = split.(data.Time, ":")
    insertcols!(data, [n => parse.(Float32, getindex.(tmp2, i)) for (i, n) in enumerate([:hour, :min, :sec])]...)
    select!(data, Not(:Time))

    return Matrix(Float32.(data))
end

"""
spgemm dataset matrix
========================
spgemm()
Return a 241600×18 matrix of Float32 values from a dataset of GPU SPGEMM kernel performance:
https://archive-beta.ics.uci.edu/ml/datasets/sgemm+gpu+kernel+performance.
"""
function spgemm()
    link = "https://archive.ics.uci.edu/ml/machine-learning-databases/00440/sgemm_product_dataset.zip"
    loc, unpack = download_dataset(link, "spgemm")
    unzip_path = dirname(loc)
    if unpack
        unzip(loc, unzip_path, true)
    end

    fname = joinpath(unzip_path, "sgemm_product.csv")
    data = readdlm(fname, ',', Float32; skipstart=1)
    return data
end
