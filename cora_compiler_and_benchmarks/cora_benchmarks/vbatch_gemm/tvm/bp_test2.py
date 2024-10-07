import numpy as np
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

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--hfuse', dest='hfuse', default=False, action='store_true')
parser.add_argument('--split', dest='split', default=False, action='store_true')

parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--batch-sizes', dest='batch_sizes', nargs='+', default=[32], type=int)
parser.add_argument('--tile-size', dest='tile_size', default=128, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
parser.add_argument('--data-file', nargs='?', default='random')
args = parser.parse_args()

# BATCH_SIZE = args.batch_size
BATCH_SIZE = te.var('bs')
HEAD_SIZE = 256
TILE=64
RTILE=4
# MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)
MAX_LEN = 128

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
sd = Dim('sd')
h1 = Dim('h1')
h2 = Dim('h2')

def lbw(name): return Ufw(name, "l", (0, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.floormult(lens[b], 64))
lbufw = lbw('lb')
if args.split:
    def ubw(name): return Ufw(name, "l", (32, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], 32))
    ubufw = ubw('ub')
else:
    def ubw(name): return Ufw(name, "l", (64, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], 64))
    ubufw = ubw('ub')

bd_uf = Uf.from_constant('bd', BATCH_SIZE, "l")
lb_uf = lbufw.get_uf()
ub_uf = ubufw.get_uf()
hd_uf = Uf.from_constant('hd', HEAD_SIZE, "l")

loop_ufs=[bd_uf, ub_uf, hd_uf]
width_ufs = None # if args.dense_storage else loop_ufs
Q = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, HEAD_SIZE), [bd, sd, h2], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[bd_uf, hd_uf, hd_uf]
width_ufs = None # if args.dense_storage else loop_ufs
K = te.ragged_placeholder((BATCH_SIZE, HEAD_SIZE, HEAD_SIZE), [bd, h1, h2], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[bd_uf, ub_uf, hd_uf]
width_ufs = [[bd_uf, Uf.from_constant('ml', MAX_LEN, "l"), hd_uf]]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, HEAD_SIZE), [bd, sd, h1], loop_ufs,
                      lambda ds: tvm.sum(Q[ds[bd], ds[sd], k] * K[ds[bd], ds[h1], k],
                                         axis = k, dimensions=[h2]),
                      name = 'O', width_uf_lists=width_ufs)

output_layout = O.op.output_layout(0)
print(output_layout)
s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")
ntx = 8
nty = 8

def schedule_op(O, tile, suffix):
    S = s.cache_write(O, 'local')

    Qs = s.cache_read(Q, "shared", [S], layouts='dense', suffix=suffix)
    Ks = s.cache_read(K, "shared", [S], layouts='dense', suffix=suffix)

    Ql = s.cache_read(Qs, "local", [S], layouts='dense', suffix=suffix)
    Kl = s.cache_read(Ks, "local", [S], layouts='dense', suffix=suffix)

    b, x, y = s[O].leaf_iter_vars[0:3]
    xo, xi = s[O].split(x, factor=tile)
    yo, yi = s[O].split(y, factor=64)

    s[O].reorder(b, xo, yo, xi, yi)
    f1 = s[O].fuse(xo, yo)
    f2 = s[O].fuse(b, f1)
    s[O].bind(f2, block_x())

    xio, xii = s[O].split(xi, factor = nty)
    yio, yii = s[O].split(yi, factor = ntx)
    s[O].bind(xii, thread_y())
    s[O].bind(yii, thread_x())
    s[O].bind(yio, tvm.thread_axis("vthread", name='vth1'), no_unroll_vthread=True)
    s[O].bind(xio, tvm.thread_axis("vthread", name='vth2'), no_unroll_vthread=True)
    s[O].reorder(xio, yii, yio, xii)
    s[S].compute_at(s[O], xii)

    x, y, k = s[S].leaf_iter_vars[1:4]
    if tile == 64:
        ko, ki = s[S].split(k, nparts=8)
        s[S].reorder(ko, ki, x, y)
        s[Qs].compute_at(s[S], ko)
        s[Ks].compute_at(s[S], ko)
    else:
        ko, ki = s[S].split(k, nparts=8)
        koo, koi = s[S].split(ko, nparts=4)
        s[S].reorder(koo, koi, ki, x, y)
        s[Qs].compute_at(s[S], koo)
        s[Ks].compute_at(s[S], koi)
    s[Ql].compute_at(s[S], ki)
    s[Kl].compute_at(s[S], ki)

    x, y = s[Ks].leaf_iter_vars[1], s[Ks].leaf_iter_vars[2]
    s[Ks].reorder(y, x)
    f = s[Ks].fuse(x, y)
    fo, fi = s[Ks].split(f, factor = ntx * nty * 4)
    fio, fii = s[Ks].split(fi, factor = ntx * 4)
    fiio, fiii = s[Ks].split(fii, factor = 4)
    s[Ks].bind(fio, thread_y())
    s[Ks].bind(fiio, thread_x())
    if not args.debug_functions: s[Ks].vectorize(fiii)

    x, y = s[Qs].leaf_iter_vars[1], s[Qs].leaf_iter_vars[2]
    s[Qs].reorder(y, x)
    f = s[Qs].fuse(x, y)
    fo, fi = s[Qs].split(f, factor = ntx * nty * 4)
    fio, fii = s[Qs].split(fi, factor = ntx * 4)
    fiio, fiii = s[Qs].split(fii, factor = 4)
    s[Qs].bind(fio, thread_y())
    s[Qs].bind(fiio, thread_x())
    if not args.debug_functions: s[Qs].vectorize(fiii)

    s.reorder_tensor_dimensions(Ks, 1, 2)
    # s.reorder_tensor_dimensions(Ks, 2, 3)
    s.reorder_tensor_dimensions(Qs, 1, 2)
    # s.reorder_tensor_dimensions(Qs, 2, 3)

if args.split:
    G1, G2 = s.split_for_bin_packing([O], O, {O.op.axis[1]: lb_uf}, include_inputs=True)
    O1, = G1
    O2, = G2
    schedule_op(O1, 64, '1')
    schedule_op(O2, 32, '2')

    if args.hfuse:
        s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0])])
else:
    schedule_op(O, 64, '1')
    # schedule_op(O, 32, '1')


gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        # Q: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (ubufw.get_fn(lens)(b))),
        # K: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (ubufw.get_fn(lens)(b))),
        # O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            # lambda b: (ubufw.get_fn(lens)(b) *
                                                       # ubufw.get_fn(lens)(b)))
    }

bO = tvm.tir.decl_buffer(output_layout, name="bO")
inputs = [[lens], [BATCH_SIZE, Q, K, bO]]
if args.split:
    binds = {O1:bO, O2:bO}
else:
    binds = {O:bO}
# inputs = [[lens], [Q, K, O]]
# binds = {}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds,
                                        run_function=run_utils.get_vbatch_gemm_run_fn(BATCH_SIZE, skip_m_k=True))
# _, Q, K, O = out
# for i in range(BATCH_SIZE):
#     length = batches[0][i]
#     rounded = utils.ceilmult(length, TILE)
#     print(rounded, np.mean(O[i,0:rounded,:,0:rounded]))
