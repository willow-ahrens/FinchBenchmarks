# using Finch
using TestImages
using ImageCore, OpenCV, TestImages, MosaicViews
using BenchmarkTools
using JSON
using ArgParse
using DataStructures



function opencv_blur(img, nthr)
    #step -1: set num Threads
    OpenCV.setNumThreads(nthr)
    # step 0: setup results
    results = Vector{Int32}([0 for _ in 1:256])
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
    (;time=time, result=result[])

end


s = ArgParseSettings("Run spgemm experiments.")

@add_arg_table! s begin
    "--output", "-o"
        arg_type = String
        help = "output file path"
        default = "hist_results.json"
    "--dataset", "-d"
        arg_type = String
        help = "dataset keyword"
        default = "test"
    "--num_iters"
        arg_type = Int
        help = "number of iters to run"
        default = 20
    "--num_threads"
        arg_type = Int
        help = "number of iters to run"
        default = 1
end

parsed_args = parse_args(ARGS, s)

num_iters = parsed_args["num_iters"]
num_threads = parsed_args["num_threads"]
@assert num_threads == Threads.nthreads()
datasets = Dict(
    "test" => [
        "Mandrill",
    ])

results = []
for img in datasets[parsed_args["dataset"]]
    imgLoad = testimage(img)
    hist_ref = nothing
    for (key, method) in [
        "opencv" => opencv_blur
    ] 
        @info "testing" key img
        res = method(imgLoad, num_threads)
        hist_ref = something(hist_ref, res.result)
        res.result == hist_ref || @warn("incorrect result")
        @info "results" res.time
        push!(results, OrderedDict(
            "time" => res.time,
            "method" => key,
            "kernel" => "hist",
            "num_threads" => num_threads,
            "image" => img,
        ))
        write(parsed_args["output"], JSON.json(results, 4))
    end
end
