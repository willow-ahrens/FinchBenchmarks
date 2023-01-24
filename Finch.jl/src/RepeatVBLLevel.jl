struct RepeatVBLLevel{Ti, Tp, RLvl, Lvl}
    I::Ti
    pos::Vector{Tp}
    idx::Vector{Ti}
    ofs::Vector{Tp}
    lvl::Lvl
    rlvl::RLvl
end
const RepeatVBL = RepeatVBLLevel
RepeatVBLLevel(lvl, rlvl) = RepeatVBLLevel(0, lvl, rlvl)
RepeatVBLLevel{Ti}(lvl, rlvl) where {Ti} = RepeatVBLLevel{Ti}(zero(Ti), lvl, rlvl)
RepeatVBLLevel{Ti, Tp}(lvl, rlvl) where {Ti, Tp} = RepeatVBLLevel{Ti, Tp}(zero(Ti), lvl, rlvl)

RepeatVBLLevel(I::Ti, lvl, rlvl) where {Ti} = RepeatVBLLevel{Ti}(I, lvl, rlvl)
RepeatVBLLevel{Ti}(I, lvl, rlvl) where {Ti} = RepeatVBLLevel{Ti, Int}(Ti(I), lvl, rlvl)
RepeatVBLLevel{Ti, Tp}(I, lvl::Lvl, rlvl::RLvl) where {Ti, Tp, Lvl, RLvl} = RepeatVBLLevel{Ti, Tp, Lvl, RLvl}(Ti(I), Ti[1, 1], Ti[], Ti[1], lvl, rlvl)

RepeatVBLLevel(I::Ti, pos::Vector{Tp}, idx, ofs, lvl::Lvl, rlvl::RLvl) where {Ti, Tp, Lvl, RLvl} = RepeatVBLLevel{Ti, Tp, Lvl, RLvl}(I, pos, idx, ofs, lvl, rlvl)
RepeatVBLLevel{Ti}(I, pos::Vector{Tp}, idx, ofs, lvl::Lvl, rlvl::RLvl) where {Ti, Tp, Lvl, RLvl} = RepeatVBLLevel{Ti, Tp, Lvl, RLvl}(Ti(I), pos, idx, ofs, lvl, rlvl)
RepeatVBLLevel{Ti, Tp}(I, pos, idx, ofs, lvl::Lvl, rlvl::RLvl) where {Ti, Tp, Lvl, RLvl} = RepeatVBLLevel{Ti, Tp, Lvl, RLvl}(Ti(I), pos, idx, ofs, lvl, rlvl)

"""
`f_code(rv)` = [RepeatVBLLevel](@ref).
"""
f_code(::Val{:rv}) = RepeatVBL
summary_f_code(lvl::RepeatVBLLevel) = "rv($(summary_f_code(lvl.lvl)), $(summary_f_code(lvl.rlvl)))"
similar_level(lvl::RepeatVBLLevel) = RepeatVBL(similar_level(lvl.lvl), similar_level(lvl.rlvl))
similar_level(lvl::RepeatVBLLevel, dim, tail...) = RepeatVBL(dim, similar_level(lvl.lvl, tail...), similar_level(lvl.rlvl, tail...))

pattern!(lvl::RepeatVBLLevel{Ti}) where {Ti} = 
    RepeatVBLLevel{Ti}(lvl.I, lvl.pos, lvl.idx, lvl.ofs, pattern!(lvl.lvl), pattern!(lvl.rlvl))

function Base.show(io::IO, lvl::RepeatVBLLevel{Ti, Tp}) where {Ti, Tp}
    if get(io, :compact, false)
        print(io, "RepeatVBL(")
    else
        print(io, "RepeatVBL{$Ti, $Tp}(")
    end
    show(IOContext(io, :typeinfo=>Ti), lvl.I)
    print(io, ", ")
    if get(io, :compact, false)
        print(io, "…")
    else
        show(IOContext(io, :typeinfo=>Vector{Tp}), lvl.pos)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>Vector{Ti}), lvl.idx)
        print(io, ", ")
        show(IOContext(io, :typeinfo=>Vector{Tp}), lvl.ofs)
    end
    print(io, ", ")
    show(io, lvl.lvl),
    print(io, ", ")
    show(io, lvl.rlvl)
    print(io, ")")
end

function display_fiber(io::IO, mime::MIME"text/plain", fbr::Fiber{<:RepeatVBLLevel})
    p = envposition(fbr.env)
    crds = []
    #TODO!!!
    # for r in fbr.lvl.pos[p]:fbr.lvl.pos[p + 1] - 1
    #     i = fbr.lvl.idx[r]
    #     l = fbr.lvl.ofs[r + 1] - fbr.lvl.ofs[r]
    #     append!(crds, (i - l + 1):i)
    # end

    depth = envdepth(fbr.env)

    print_coord(io, crd) = (print(io, "["); show(io, crd); print(io, "]"))
    get_fbr(crd) = fbr(crd)

    print(io, "│ " ^ depth); print(io, "RepeatVBL ("); show(IOContext(io, :compact=>true), default(fbr)); print(io, ") ["); show(io, 1); print(io, ":"); show(io, fbr.lvl.I); println(io, "]")
    display_fiber_data(io, mime, fbr, 1, crds, print_coord, get_fbr)
end

@inline Base.ndims(fbr::Fiber{<:RepeatVBLLevel}) = 1 + ndims(Fiber(fbr.lvl.lvl, Environment(fbr.env)))
@inline Base.size(fbr::Fiber{<:RepeatVBLLevel}) = (fbr.lvl.I, size(Fiber(fbr.lvl.lvl, Environment(fbr.env)))...)
@inline Base.axes(fbr::Fiber{<:RepeatVBLLevel}) = (1:fbr.lvl.I, axes(Fiber(fbr.lvl.lvl, Environment(fbr.env)))...)
@inline Base.eltype(fbr::Fiber{<:RepeatVBLLevel}) = eltype(Fiber(fbr.lvl.lvl, Environment(fbr.env)))
@inline default(fbr::Fiber{<:RepeatVBLLevel}) = default(Fiber(fbr.lvl.lvl, Environment(fbr.env)))

(fbr::Fiber{<:RepeatVBLLevel})() = fbr
function (fbr::Fiber{<:RepeatVBLLevel})(i, tail...)
    # TODO
    exit(1)
    # lvl = fbr.lvl
    # p = envposition(fbr.env)
    # r = lvl.pos[p] + searchsortedfirst(@view(lvl.idx[lvl.pos[p]:lvl.pos[p + 1] - 1]), i) - 1
    # r < lvl.pos[p + 1] || return default(fbr)
    # q = lvl.ofs[r + 1] - 1 - lvl.idx[r] + i
    # q >= lvl.ofs[r] || return default(fbr)
    # fbr_2 = Fiber(lvl.lvl, Environment(position=q, index=i, parent=fbr.env))
    # return fbr_2(tail...)
end

mutable struct VirtualRepeatVBLLevel
    ex
    Ti
    Tp
    I
    pos_alloc
    pos_fill
    pos_stop
    idx_alloc
    ofs_alloc
    lvl
    rlvl
end
function virtualize(ex, ::Type{RepeatVBLLevel{Ti, Tp, Lvl, RLvl}}, ctx, tag=:lvl) where {Ti, Tp, Lvl, RLvl}
    sym = ctx.freshen(tag)
    I = value(:($sym.I), Int)
    pos_alloc = ctx.freshen(sym, :_pos_alloc)
    pos_fill = ctx.freshen(sym, :_pos_fill)
    pos_stop = ctx.freshen(sym, :_pos_stop)
    idx_alloc = ctx.freshen(sym, :_idx_alloc)
    ofs_alloc = ctx.freshen(sym, :_ofs_alloc)
    push!(ctx.preamble, quote
        $sym = $ex
        $pos_alloc = length($sym.pos)
        $idx_alloc = length($sym.idx)
        $ofs_alloc = length($sym.ofs)
    end)
    lvl_2 = virtualize(:($sym.lvl), Lvl, ctx, sym)
    rlvl_2 = virtualize(:($sym.rlvl), RLvl, ctx, sym)
    VirtualRepeatVBLLevel(sym, Ti, Tp, I, pos_alloc, pos_fill, pos_stop, idx_alloc, ofs_alloc, lvl_2, rlvl_2)
end
function (ctx::Finch.LowerJulia)(lvl::VirtualRepeatVBLLevel)
    quote
        $RepeatVBLLevel{$(lvl.Ti)}(
            $(ctx(lvl.I)),
            $(lvl.ex).pos,
            $(lvl.ex).idx,
            $(lvl.ex).ofs,
            $(ctx(lvl.lvl)),
            $(ctx(lvl.rlvl)),
        )
    end
end

summary_f_code(lvl::VirtualRepeatVBLLevel) = "rv($(summary_f_code(lvl.lvl)), $(summary_f_code(lvl.rlvl)))"

hasdefaultcheck(lvl::VirtualRepeatVBLLevel) = true

getsites(fbr::VirtualFiber{VirtualRepeatVBLLevel}) =
    [envdepth(fbr.env) + 1, getsites(VirtualFiber(fbr.lvl.lvl, VirtualEnvironment(fbr.env)))...]

function getsize(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx, mode)
    ext = Extent(literal(fbr.lvl.Ti(1)), fbr.lvl.I)
    if mode.kind !== reader
        ext = suggest(ext)
    end
    (ext, getsize(VirtualFiber(fbr.lvl.lvl, VirtualEnvironment(fbr.env)), ctx, mode)...)
end

function setsize!(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx, mode, dim, dims...)
    fbr.lvl.I = getstop(dim)
    fbr.lvl.lvl = setsize!(VirtualFiber(fbr.lvl.lvl, VirtualEnvironment(fbr.env)), ctx, mode, dims...).lvl
    fbr
end

@inline default(fbr::VirtualFiber{<:VirtualRepeatVBLLevel}) = default(VirtualFiber(fbr.lvl.lvl, VirtualEnvironment(fbr.env)))
Base.eltype(fbr::VirtualFiber{VirtualRepeatVBLLevel}) = eltype(VirtualFiber(fbr.lvl.lvl, VirtualEnvironment(fbr.env)))

function initialize_level!(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx::LowerJulia, mode)
    lvl = fbr.lvl
    Tp = lvl.Tp
    Ti = lvl.Ti
    push!(ctx.preamble, quote
        $(lvl.pos_alloc) = length($(lvl.ex).pos)
        $(lvl.ex).pos[1] = $(Tp(1))
        $(lvl.ex).pos[2] = $(Tp(1))
        $(lvl.pos_fill) = 1
        $(lvl.pos_stop) = 2
        $(lvl.ofs_alloc) = length($(lvl.ex).ofs)
        $(lvl.ex).ofs[1] = $(Tp(1))
        $(lvl.idx_alloc) = length($(lvl.ex).idx)
    end)
    lvl.lvl = initialize_level!(VirtualFiber(fbr.lvl.lvl, Environment(fbr.env)), ctx, mode)
    lvl.rlvl = initialize_level!(VirtualFiber(fbr.lvl.rlvl, Environment(fbr.env)), ctx, mode)
    return lvl
end

function trim_level!(lvl::VirtualRepeatVBLLevel, ctx::LowerJulia, pos)
    Tp = lvl.Tp
    Ti = lvl.Ti
    qos = ctx.freshen(:qos)
    push!(ctx.preamble, quote
        $(lvl.pos_alloc) = $(ctx(pos)) + 1
        resize!($(lvl.ex).pos, $(lvl.pos_alloc))
        $(lvl.idx_alloc) = $(lvl.ex).pos[$(lvl.pos_alloc)] - 1
        resize!($(lvl.ex).idx, $(lvl.idx_alloc))
        $(lvl.ofs_alloc) = $(lvl.idx_alloc) + 1
        resize!($(lvl.ex).ofs, $(lvl.ofs_alloc))
        $(qos) = $(lvl.ex).ofs[$(lvl.ofs_alloc)] - $(Tp(1))
    end)
    lvl.lvl = trim_level!(lvl.lvl, ctx, qos)
    lvl.rlvl = trim_level!(lvl.rlvl, ctx, qos)
    return lvl
end

interval_assembly_depth(lvl::VirtualRepeatVBLLevel) = Inf

#This function is quite simple, since RepeatVBLLevels don't support reassembly.
function assemble!(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx, mode)
    lvl = fbr.lvl
    p_stop = ctx(cache!(ctx, ctx.freshen(lvl.ex, :_p_stop), getstop(envposition(fbr.env))))
    push!(ctx.preamble, quote
        $(lvl.pos_stop) = $p_stop + 1
        $Finch.@regrow!($(lvl.ex).pos, $(lvl.pos_alloc), $(lvl.pos_stop))
    end)
end

function finalize_level!(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx::LowerJulia, mode)
    lvl = fbr.lvl
    my_p = ctx.freshen(:p)
    my_q = ctx.freshen(:q)
    push!(ctx.preamble, quote
        $my_q = $(lvl.ex).pos[$(lvl.pos_fill)]
        for $my_p = $(lvl.pos_fill):$(lvl.pos_stop)
            $(lvl.ex).pos[$(my_p)] = $my_q
        end
    end)
    fbr.lvl.lvl = finalize_level!(VirtualFiber(fbr.lvl.lvl, VirtualEnvironment(fbr.env)), ctx, mode)
    fbr.lvl.rlvl = finalize_level!(VirtualFiber(fbr.lvl.rlvl, VirtualEnvironment(fbr.env)), ctx, mode)
    return fbr.lvl
end

function unfurl(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx, mode, ::Nothing, idx, idxs...)
    if idx.kind === protocol
        @assert idx.mode.kind === literal
        unfurl(fbr, ctx, mode, idx.mode.val, idx.idx, idxs...)
    elseif mode.kind === reader
        unfurl(fbr, ctx, mode, walk, idx, idxs...)
    else
        unfurl(fbr, ctx, mode, extrude, idx, idxs...)
    end
end

function unfurl(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx, mode, ::Walk, idx, idxs...)
    lvl = fbr.lvl
    tag = lvl.ex
    Tp = lvl.Tp
    Ti = lvl.Ti
    my_i = ctx.freshen(tag, :_i)
    my_i_start = ctx.freshen(tag, :_i)
    my_r = ctx.freshen(tag, :_r)
    my_r_stop = ctx.freshen(tag, :_r_stop)
    my_q = ctx.freshen(tag, :_q)
    my_q_stop = ctx.freshen(tag, :_q_stop)
    my_q_ofs = ctx.freshen(tag, :_q_ofs)
    my_i1 = ctx.freshen(tag, :_i1)
    
    run_refurl = refurl(VirtualFiber(lvl.rlvl, VirtualEnvironment(position=value(my_r, lvl.Ti), index=my_i_start, parent=fbr.env)), ctx, mode)

    body = Thunk(
        preamble = quote
            $my_r = $(lvl.ex).pos[$(ctx(envposition(fbr.env)))]
            $my_r_stop = $(lvl.ex).pos[$(ctx(envposition(fbr.env))) + $(Tp(1))]
            if $my_r < $my_r_stop
                $my_i = $(lvl.ex).idx[$my_r]
                $my_i1 = $(lvl.ex).idx[$my_r_stop - $(Tp(1))]
            else
                $my_i = $(Ti(1))
                $my_i1 = $(Ti(0))
            end
        end,
        body = Pipeline([
            Phase(
                stride = (ctx, idx, ext) -> value(my_i1),
                body = (start, step) -> Stepper(
                    seek = (ctx, ext) -> quote
                        while $my_r + $(Tp(1)) < $my_r_stop && $(lvl.ex).idx[$my_r] < $(ctx(getstart(ext)))
                            $my_r += $(Tp(1))
                        end
                    end,
                    body = Thunk(
                        preamble = quote
                            $my_i = $(lvl.ex).idx[$my_r]
                            $my_q_stop = $(lvl.ex).ofs[$my_r + $(Tp(1))]
                            $my_i_start = $my_i - ($my_q_stop - $(lvl.ex).ofs[$my_r])
                            $my_q_ofs = $my_q_stop - $my_i - $(Tp(1))
                        end,
                        body = Step(
                            stride = (ctx, idx, ext) -> value(my_i),
                            body = (ctx, idx, ext, ext_2) -> Thunk(
                                body = Pipeline([
                                    Phase(
                                        stride = (ctx, idx, ext) -> value(my_i_start),
                                        # body = (start, step) -> Run(Simplify(Fill(default(fbr)))),
                                        body = (start, step) -> Thunk(
                                            preamble = run_refurl.preamble,
                                            body = Switch([
                                                value(:($(run_refurl.body.lvl.val) == $(default(fbr)))) => Simplify(Fill(default(fbr))),
                                                literal(true) => Run(body = run_refurl)
                                            ])
                                            # body = Run(
                                            #     # Simplify(Fill($my_tmp_val))
                                            #     # body = refurl(VirtualFiber(lvl.rlvl, VirtualEnvironment(position=value(my_r, lvl.Ti), index=my_i_start, parent=fbr.env)), ctx, mode)
                                            # )
                                        ),
                                    ),
                                    Phase(
                                        body = (start, step) -> Lookup(
                                            body = (i) -> Thunk(
                                                preamble = quote
                                                    $my_q = $my_q_ofs + $(ctx(i))
                                                end,
                                                body = refurl(VirtualFiber(lvl.lvl, VirtualEnvironment(position=value(my_q, lvl.Ti), index=i, parent=fbr.env)), ctx, mode),
                                            )
                                        )
                                    )
                                ]),
                                epilogue = quote
                                    $my_r += ($(ctx(getstop(ext_2))) == $my_i)
                                end
                            )
                        )
                    )
                )
            ),
            Phase(
                # body = (start, step) -> Run(Simplify(Fill(default(fbr))))
                body = (start, step) -> Thunk(
                    preamble = quote
                        # 325
                    end,
                    body = Run(
                        # Simplify(Fill($my_tmp_val))
                        body = refurl(VirtualFiber(lvl.rlvl, VirtualEnvironment(position=value(my_r_stop, lvl.Ti), index=my_i1, parent=fbr.env)), ctx, mode)

                    )
                )
            )
        ])
    )

    exfurl(body, ctx, mode, idx)
end

#ex: [1,1,5,5,6,7,8] -> runs[1,5,0], vals[6,7,8], idx[2,4,7], ofs[1,1,4]
function to_fiber_rvb(vec::Vector{Tv}, Ti=Int, Tp=Int) where {Tv}
    I = length(vec)
    idx = Vector{Ti}()
    ofs = [Tp(1)]

    vals = Vector{Tv}()
    runs = Vector{Tv}()

    i = 1
    while i <= I
        run_start = i
        while i <= I && vec[i] == vec[run_start]
           i+=1
        end
        append!(runs, vec[run_start])
        if i == run_start +1
            i = run_start
        end

        while i+1 <= I && vec[i] != vec[i+1]
            append!(vals, vec[i])
            i+=1
        end
        if i == I
            append!(vals, vec[i])
            i+=1
        end
        append!(idx, i-1)
        append!(ofs, length(vals) + 1)
    end

    pos = [Tp(1), Tp(length(idx) + 1)]
    @fiber RepeatVBL{Ti, Tp}(I, pos, idx, ofs, Element{0, Tv}(vals), Element{0, Tv}(runs))
end


function to_fiber_mtx_rvb(mtx::Matrix{Tv}, Ti=Int, Tp=Int) where {Tv}
    J = size(mtx, 2)
    idx = Vector{Ti}()
    ofs = [Tp(1)]
    pos = [Tp(1)]

    vals = Vector{Tv}()
    runs = Vector{Tv}()

    for i in 1:size(mtx, 1)
        j = 1
        while j <= J
            run_start = j
            while j <= J && mtx[i,j] == mtx[i,run_start]
            j+=1
            end
            append!(runs, mtx[i,run_start])
            if j == run_start +1
                j = run_start
            end

            while j+1 <= J && mtx[i,j] != mtx[i,j+1]
                append!(vals, mtx[i,j])
                j+=1
            end
            if j == J
                append!(vals, mtx[i,j])
                j+=1
            end
            append!(idx, j-1)
            append!(ofs, length(vals) + 1)
        end
        append!(pos, Tp(length(idx) + 1))
    end

    @fiber Dense{Ti}(size(mtx, 1), RepeatVBL{Ti, Tp}(J, pos, idx, ofs, Element{0, Tv}(vals), Element{0, Tv}(runs)))
end

# function unfurl(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx, mode, ::Extrude, idx, idxs...)
#     lvl = fbr.lvl
#     tag = lvl.ex
#     Tp = lvl.Tp
#     Ti = lvl.Ti
#     my_p = ctx.freshen(tag, :_p)
#     my_q = ctx.freshen(tag, :_q)
#     my_i_prev = ctx.freshen(tag, :_i_prev)
#     my_r = ctx.freshen(tag, :_r)
#     my_guard = if hasdefaultcheck(lvl.lvl)
#         ctx.freshen(tag, :_isdefault)
#     end

#     push!(ctx.preamble, quote
#         $my_r = $(lvl.ex).pos[$(lvl.pos_fill)]
#         $my_q = $(lvl.ex).ofs[$my_r]
#         for $my_p = $(lvl.pos_fill):$(ctx(envposition(fbr.env)))
#             $(lvl.ex).pos[$(my_p)] = $my_r
#         end
#         $my_i_prev = $(Ti(-1))
#     end)

#     body = AcceptSpike(
#         val = default(fbr),
#         tail = (ctx, idx) -> Thunk(
#             preamble = quote
#                 $(begin
#                     assemble!(VirtualFiber(lvl.lvl, VirtualEnvironment(position=value(my_q, lvl.Ti), parent=fbr.env)), ctx, mode)
#                     quote end
#                 end)
#                 $(
#                     if hasdefaultcheck(lvl.lvl)
#                         :($my_guard = true)
#                     else
#                         quote end
#                     end
#                 )
#             end,
#             body = refurl(VirtualFiber(lvl.lvl, VirtualEnvironment(position=value(my_q, lvl.Ti), index=idx, guard=my_guard, parent=fbr.env)), ctx, mode),
#             epilogue = begin
#                 #We should be careful here. Presumably, we haven't modified the subfiber because it is still default. Is this always true? Should strict assembly happen every time?
#                 body = quote
#                     if $(ctx(idx)) > $my_i_prev + $(Ti(1))
#                         $Finch.@regrow!($(lvl.ex).idx, $(lvl.idx_alloc), $my_r)
#                         $Finch.@regrow!($(lvl.ex).ofs, $(lvl.ofs_alloc), $my_r + $(Tp(1)))
#                         $my_r += $(Tp(1))
#                     end
#                     $(lvl.ex).idx[$my_r - $(Tp(1))] = $my_i_prev = $(ctx(idx))
#                     $(my_q) += $(Tp(1))
#                     $(lvl.ex).ofs[$my_r] = $my_q
#                 end
#                 if envdefaultcheck(fbr.env) !== nothing
#                     body = quote
#                         $body
#                         $(envdefaultcheck(fbr.env)) = false
#                     end
#                 end
#                 if hasdefaultcheck(lvl.lvl)
#                     body = quote
#                         if !$(my_guard)
#                             $body
#                         end
#                     end
#                 end
#                 body
#             end
#         )
#     )

#     push!(ctx.epilogue, quote
#         $(lvl.ex).pos[$(ctx(envposition(fbr.env))) + $(Tp(1))] = $my_r
#         $(lvl.pos_fill) = $(ctx(envposition(fbr.env))) + $(Tp(1))
#     end)

#     exfurl(body, ctx, mode, idx)
# end

# function unfurl(fbr::VirtualFiber{VirtualRepeatVBLLevel}, ctx, mode, ::Extrude, idx, idxs...)
#     lvl = fbr.lvl
#     tag = lvl.ex
#     Tp = lvl.Tp
#     Ti = lvl.Ti
#     my_p = ctx.freshen(tag, :_p)
#     my_q = ctx.freshen(tag, :_q)
#     my_i_prev = ctx.freshen(tag, :_i_prev)
#     my_r = ctx.freshen(tag, :_r)
#     my_guard = if hasdefaultcheck(lvl.lvl)
#         ctx.freshen(tag, :_isdefault)
#     end

#     function record_run(ctx, stop, v)
#         quote
#             $(lvl.idx_alloc) < $my_q && ($(lvl.idx_alloc) = $Finch.regrow!($(lvl.ex).idx, $(lvl.idx_alloc), $my_q))
#             $(lvl.val_alloc) < $my_q && ($(lvl.val_alloc) = $Finch.regrow!($(lvl.ex).val, $(lvl.val_alloc), $my_q))
#             $(lvl.ex).idx[$my_q] = $(ctx(stop))
#             $(lvl.ex).val[$my_q] = $v
#             $my_q += $(Tp(1))
#         end
#     end
    

#     push!(ctx.preamble, quote
#         $my_r = $(lvl.ex).pos[$(lvl.pos_fill)]
#         $my_q = $(lvl.ex).ofs[$my_r]
#         for $my_p = $(lvl.pos_fill):$(ctx(envposition(fbr.env)))
#             $(lvl.ex).pos[$(my_p)] = $my_r
#         end
#         $my_i_prev = $(Ti(-1))
#     end)

# end
