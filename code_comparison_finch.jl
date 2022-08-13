using Finch

using Finch: LowerJulia, getname, getstart, getstop
using Finch.IndexNotation
using SyntaxInterface
using Finch: Run, Spike, Extent, Scalar, Cases, Stepper, Jumper, Step, Jump, AcceptRun, AcceptSpike, Thunk, Phase, Pipeline, Leaf, Simplify, Shift

mutable struct SingleBlock{D, Tv, Ti} <: AbstractVector{Tv}
    len::Ti
    start::Ti
    stop::Ti
    val::Vector{Tv}
end

function SingleBlock{D}(len::Ti, start::Ti, stop::Ti, val::Vector{Tv}) where {D, Ti, Tv}
    SingleBlock{D, Tv, Ti}(len, start, stop, val)
end

Base.size(vec::SingleBlock) = (vec.len,)

function Base.getindex(vec::SingleBlock{D, Tv, Ti}, i) where {D, Tv, Ti}
    if i < vec.start
        D
    elseif i <= vec.stop
        vec.val[i - vec.start + 1]
    else
        D
    end
end

mutable struct VirtualSingleBlock{Tv, Ti}
    ex
    name
    D
end

function Finch.virtualize(ex, ::Type{SingleBlock{D, Tv, Ti}}, ctx, tag=:tns) where {D, Tv, Ti}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualSingleBlock{Tv, Ti}(sym, tag, D)
end

(ctx::Finch.LowerJulia)(tns::VirtualSingleBlock) = tns.ex

function Finch.getdims(arr::VirtualSingleBlock{Tv, Ti}, ctx::Finch.LowerJulia, mode) where {Tv, Ti}
    ex = Symbol(arr.name, :_stop)
    push!(ctx.preamble, :($ex = $size($(arr.ex))[1]))
    (Extent(1, Virtual{Ti}(ex)),)
end
Finch.setdims!(arr::VirtualSingleBlock{Tv, Ti}, ctx::Finch.LowerJulia, mode, dims...) where {Tv, Ti} = arr
Finch.getname(arr::VirtualSingleBlock) = arr.name
Finch.setname(arr::VirtualSingleBlock, name) = (arr_2 = deepcopy(arr); arr_2.name = name; arr_2)
function (ctx::Finch.Stylize{LowerJulia})(node::Access{<:VirtualSingleBlock})
    if ctx.root isa Loop && ctx.root.idx == Finch.get_furl_root(node.idxs[1])
        Finch.ChunkStyle()
    else
        mapreduce(ctx, Finch.result_style, arguments(node))
    end
end

function (ctx::Finch.ChunkifyVisitor)(node::Access{VirtualSingleBlock{Tv, Ti}, Read}, ::Finch.DefaultStyle) where {Tv, Ti}
    vec = node.tns
    if getname(ctx.idx) == getname(node.idxs[1])
        tns = Pipeline([
            Phase(
                stride = (ctx, idx, ext) -> :($(vec.ex).start - 1),
                body = (start, step) -> Run(body = Simplify(vec.D))
            ),
            Phase(
                stride = (ctx, idx, ext) -> :($(vec.ex).stop),
                body = (start, step) -> Leaf(
                    body = (i) -> :($(vec.ex).val[$(ctx.ctx(i)) - $(vec.ex).start + 1]) #TODO all of these functions should really have a ctx
                )
            ),
            Phase(body = (start, step) -> Run(body = Simplify(vec.D)))
        ])
        Access(tns, node.mode, node.idxs)
    else
        node
    end
end

mutable struct SimpleSparseVector{D, Tv, Ti} <: AbstractVector{Tv}
    len::Ti
    idx::Vector{Ti}
    val::Vector{Tv}
end

function SimpleSparseVector{D}(idx::Vector{Ti}, val::Vector{Tv}) where {D, Ti, Tv}
    SimpleSparseVector{D, Tv, Ti}(idx, val)
end

Base.size(vec::SimpleSparseVector) = (vec.len,)

function Base.getindex(vec::SimpleSparseVector{D, Tv, Ti}, i) where {D, Tv, Ti}
    p = findfirst(j->j >= i, vec.idx)
    vec.idx[p] == i ? vec.val[p] : D
end

mutable struct VirtualSimpleSparseVector{Tv, Ti}
    ex
    name
    D
end

Finch.default(vec::VirtualSimpleSparseVector) = vec.D

function Finch.virtualize(ex, ::Type{SimpleSparseVector{D, Tv, Ti}}, ctx, tag=:tns) where {D, Tv, Ti}
    sym = ctx.freshen(tag)
    push!(ctx.preamble, :($sym = $ex))
    VirtualSimpleSparseVector{Tv, Ti}(sym, tag, D)
end

(ctx::Finch.LowerJulia)(tns::VirtualSimpleSparseVector) = tns.ex

function Finch.getdims(arr::VirtualSimpleSparseVector{Tv, Ti}, ctx::Finch.LowerJulia, mode) where {Tv, Ti}
    ex = Symbol(arr.name, :_stop)
    push!(ctx.preamble, :($ex = $size($(arr.ex))[1]))
    (Extent(1, Virtual{Ti}(ex)),)
end
Finch.setdims!(arr::VirtualSimpleSparseVector{Tv, Ti}, ctx::Finch.LowerJulia, mode, dims...) where {Tv, Ti} = arr
Finch.getname(arr::VirtualSimpleSparseVector) = arr.name
Finch.setname(arr::VirtualSimpleSparseVector, name) = (arr_2 = deepcopy(arr); arr_2.name = name; arr_2)
function (ctx::Finch.Stylize{LowerJulia})(node::Access{<:VirtualSimpleSparseVector})
    if ctx.root isa Loop && ctx.root.idx == Finch.get_furl_root(node.idxs[1])
        Finch.ChunkStyle()
    else
        mapreduce(ctx, Finch.result_style, arguments(node))
    end
end

function (ctx::Finch.ChunkifyVisitor)(node::Access{VirtualSimpleSparseVector{Tv, Ti}, Read}, ::Finch.DefaultStyle) where {Tv, Ti}
    vec = node.tns
    my_i = ctx.ctx.freshen(getname(vec), :_i0)
    my_i′ = ctx.ctx.freshen(getname(vec), :_i1)
    my_p = ctx.ctx.freshen(getname(vec), :_p)
    if getname(ctx.idx) == getname(node.idxs[1])
        tns = Thunk(
            preamble = quote
                $my_p = 1
                $my_i = 1
                $my_i′ = $(vec.ex).idx[$my_p]
            end,
            body = Pipeline([
                Phase(
                    stride = (ctx, idx, ext) -> :($(vec.ex).idx[end]),
                    body = (start, stop) -> Stepper(
                        seek = (ctx, ext) -> quote
                            $my_p = searchsortedfirst($(vec.ex).idx, $(ctx(getstart(ext))), $my_p, length($(vec.ex).idx), Base.Forward)
                            $my_i = $(ctx(getstart(ext)))
                            $my_i′ = $(vec.ex).idx[$my_p]
                        end,
                        body = Step(
                            stride = (ctx, idx, ext) -> my_i′,
                            chunk = Spike(
                                body = Simplify(vec.D),
                                tail = Virtual{Tv}(:($(vec.ex).val[$my_p])),
                            ),
                            next = (ctx, idx, ext) -> quote
                                $my_p += 1
                                $my_i = $my_i′ + 1
                                $my_i′ = $(vec.ex).idx[$my_p]
                            end
                        )
                    )
                ),
                Phase(body = (start, step) -> Run(body = Simplify(vec.D)))
            ])
        )
        Access(tns, node.mode, node.idxs)
    else
        node
    end
end

Finch.register()



A = SimpleSparseVector{0.0, Float64, Int}(10, [1, 3, 5, 7, 9, 11], [2.0, 3.0, 4.0, 5.0, 6.0])
B = SingleBlock{0.0, Float64, Int}(10, 3, 9, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
C = Scalar{0.0}()

println(@index_code @loop i C[] += A[i] * B[i])
@index @loop i C[] += A[i] * B[i]
println(C)