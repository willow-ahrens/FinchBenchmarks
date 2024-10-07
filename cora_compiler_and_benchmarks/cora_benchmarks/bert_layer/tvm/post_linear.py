import numpy as np
import math
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils

parser = run_utils.get_cmd_parser()
parser.add_argument('--no-hoist-loads', dest='no_hoist_loads', default=False, action='store_true')
args = parser.parse_args()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
HEAD_SIZE = 64
OUT_SIZE = 512

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
hd = Dim('hd')
od = Dim('od')
mdhd = Dim('mdhd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufw1.get_uf(),
    3: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    4: Uf.from_constant('od', OUT_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
if args.layout_unfused:
    width_ufs=[ls[0], lufw1.get_uf(), ls[1], ls[3]]
else:
    width_ufs=[ls[0], lufw64.get_uf(), ls[1], ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s1, md, hd], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((NUM_HEADS * HEAD_SIZE, OUT_SIZE), name='W')
B = te.placeholder((OUT_SIZE,), name='B')

loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=loop_ufs
A2 = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                           name='A2', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, NUM_HEADS * HEAD_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(A[ds[bd], ds[s1], tvm.floordiv(k, HEAD_SIZE), tvm.floormod(k, HEAD_SIZE)] *
                                         W[k, ds[od]], axis=k, dimensions = [mdhd]),
                      name = 'S', width_uf_lists=width_ufs)

def compute_body(ds):
    if args.skip_residual: return S[ds[bd], ds[s1], ds[od]] + B[ds[od]]
    else: return A2[ds[bd], ds[s1], ds[od]] + S[ds[bd], ds[s1], ds[od]] + B[ds[od]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      compute_body, name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    if False:
        tile = 128
        rtile = 8
        nt = tile // rtile
        ks = utils.next_power_of_2((NUM_HEADS * HEAD_SIZE) / (6144 // tile))

        thread_x = lambda: tvm.thread_axis("threadIdx.x")
        thread_y = lambda: tvm.thread_axis("threadIdx.y")
        block_x = lambda: tvm.thread_axis("blockIdx.x")
        block_y = lambda: tvm.thread_axis("blockIdx.y")
        vthread = lambda: tvm.thread_axis("vthread")

        As = s.cache_read(A, "shared", [S], loop_layout=[ls[0], ls[2], ls[1], ls[3]], layouts=[ls[0], ls[2], ls[1], ls[3]])
        Ws = s.cache_read(W, "shared", [S], vanilla=True)
        Bs = s.cache_read(B, "shared", [O], vanilla=True)

        Al = s.cache_read(As, "local", [S])
        Wl = s.cache_read(Ws, "local", [S], vanilla=True)

        b, l, h = s[O].leaf_iter_vars
        y = s[O].fuse(b, l, padding = tile)
        x = h
        yo, yi = s[O].split(y, factor = tile)
        xo, xi = s[O].split(x, factor = tile)
        s[O].bind(yo, block_y())
        s[O].bind(xo, block_x())

        yio, yii = s[O].split(yi, factor = nt)
        xio, xii = s[O].split(xi, factor = nt)
        s[O].bind(xii, thread_x())
        s[O].bind(yii, thread_y())
        s[O].bind(xio, tvm.thread_axis("vthread", name='vth1'), no_unroll_vthread = True)
        s[O].bind(yio, tvm.thread_axis("vthread", name='vth2'), no_unroll_vthread = True)
        s[S].compute_at(s[O], xii)
        s[Bs].compute_at(s[O], xii)

        b, x, y, k = s[S].leaf_iter_vars
        s[S].reorder(k, x, y)
        ko, ki = s[S].split(k, nparts = ks)
        s[As].compute_at(s[S], ko)
        s[Ws].compute_at(s[S], ko)
        s[Al].compute_at(s[S], ki)
        s[Wl].compute_at(s[S], ki)

        b, l, h, i = s[As].leaf_iter_vars
        s[As].reorder(h, b, l)
        f = s[As].fuse(b, l)
        f = s[As].fuse(f, i)
        fo, fi = s[As].split(f, factor = nt * nt * 4)
        fio, fii = s[As].split(fi, factor = nt * 4)
        fiio, fiii = s[As].split(fii, factor = 4)
        s[As].bind(fio, thread_y())
        s[As].bind(fiio, thread_x())
        if not args.debug_functions: s[As].vectorize(fiii)

        s.fuse_tensor_dimensions(As, 0, 1)

        s[S].set_scope('local')

        x, y = s[Ws].leaf_iter_vars
        f = s[Ws].fuse(x, y)
        fo, fi = s[Ws].split(f, factor = nt * nt * 4)
        fio, fii = s[Ws].split(fi, factor = nt * 4)
        fiio, fiii = s[Ws].split(fii, factor = 4)
        s[Ws].bind(fio, thread_y())
        s[Ws].bind(fiio, thread_x())
        if not args.debug_functions: s[Ws].vectorize(fiii)

        x, = s[Bs].leaf_iter_vars
        fo, fi = s[Bs].split(x, factor = nt * nt)
        fio, fii = s[Bs].split(fi, factor = nt)
        s[Bs].bind(fio, thread_y())
        s[Bs].bind(fii, thread_x())

        gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
        _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
        _ = tvm.register_func(
            utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
    else:
        tile = 64
        rtile = 8
        nt = 8
        ks = 64

        thread_x = lambda: tvm.thread_axis("threadIdx.x")
        thread_y = lambda: tvm.thread_axis("threadIdx.y")
        block_x = lambda: tvm.thread_axis("blockIdx.x")
        block_y = lambda: tvm.thread_axis("blockIdx.y")

        As = s.cache_read(A, "shared", [S], loop_layout=[ls[0], ls[2], ls[1], ls[3]], layouts=[ls[0], ls[2], ls[1], ls[3]])
        Ws = s.cache_read(W, "shared", [S], vanilla=True)
        Bs = s.cache_read(B, "shared", [O], vanilla=True)

        Al = s.cache_read(As, "local", [S])
        Wl = s.cache_read(Ws, "local", [S], vanilla=True)

        b, l, h = s[O].leaf_iter_vars
        y = s[O].fuse(b, l, padding = tile)
        x = h
        yo, yi = s[O].split(y, factor = tile)
        xo, xi = s[O].split(x, factor = tile)
        s[O].bind(yo, block_y())
        s[O].bind(xo, block_x())

        yio, yii = s[O].split(yi, factor = nt)
        xio, xii = s[O].split(xi, factor = nt)
        s[O].bind(xii, thread_x())
        s[O].bind(yii, thread_y())
        s[O].bind(xio, tvm.thread_axis("vthread", name='vth1'), no_unroll_vthread = True)
        s[O].bind(yio, tvm.thread_axis("vthread", name='vth2'), no_unroll_vthread = False)
        s[S].compute_at(s[O], xii)
        s[Bs].compute_at(s[O], xii)

        b, x, y, k = s[S].leaf_iter_vars
        s[S].reorder(k, x, y)
        ko, ki = s[S].split(k, nparts = ks)
        s[As].compute_at(s[S], ko)
        s[Ws].compute_at(s[S], ko)
        s[Al].compute_at(s[S], ki)
        s[Wl].compute_at(s[S], ki)

        b, l, h, i = s[As].leaf_iter_vars
        s[As].reorder(h, b, l)
        f = s[As].fuse(b, l)
        f = s[As].fuse(f, i)
        fo, fi = s[As].split(f, factor = nt * nt * 4)
        fio, fii = s[As].split(fi, factor = nt * 4)
        fiio, fiii = s[As].split(fii, factor = 4)
        s[As].bind(fio, thread_y())
        s[As].bind(fiio, thread_x())
        s[As].mark_no_bounds_check()
        if not args.debug_functions: s[As].vectorize(fiii)

        s.fuse_tensor_dimensions(As, 0, 1)

        s[S].set_scope('local')

        # x, y = s[Ws].leaf_iter_vars
        # f = s[Ws].fuse(x, y)
        # fo, fi = s[Ws].split(f, factor = nt * nt * 4)
        # fio, fii = s[Ws].split(fi, factor = nt * 4)
        # fiio, fiii = s[Ws].split(fii, factor = 4)
        # s[Ws].bind(fio, thread_y())
        # s[Ws].bind(fiio, thread_x())
        # if not args.debug_functions: s[Ws].vectorize(fiii)

        x, y = s[Ws].leaf_iter_vars
        s[Ws].bind(x, thread_y())
        fio, fii = s[Ws].split(y, factor = nt * 4)
        fiio, fiii = s[Ws].split(fii, factor = 4)
        s[Ws].bind(fiio, thread_x())
        s[Ws].vectorize(fiii)

        x, = s[Bs].leaf_iter_vars
        fo, fi = s[Bs].split(x, factor = nt * nt)
        fio, fii = s[Bs].split(fi, factor = nt)
        s[Bs].bind(fio, thread_y())
        s[Bs].bind(fii, thread_x())

        gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
        _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
        _ = tvm.register_func(
            utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
    if args.skip_residual:
        inputs = [[lens], [BS_VAR, A, W, B, O]]
    else:
        inputs = [[lens], [BS_VAR, A, A2, W, B, O]]
else:
    tile = 64
    if args.skip_residual:
        inputs = [[lens], [BS_VAR, A, W, B, S, O]]
    else:
        inputs = [[lens], [BS_VAR, A, A2, W, B, S, O]]

def size_fn(l_inputs):
    lens = l_inputs[0]

    if args.dense_storage: return {}
    return {
        A: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(
            len(lens), lambda b: (lufw1 if args.layout_unfused else lufw64).get_fn(lens)(b)),
        A2: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                        run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b))),
        O: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                       run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)))
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=tile,
                                        hoist_loads=not args.no_hoist_loads,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

# O = out[-1]
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]))
#     ctr += this_extent
