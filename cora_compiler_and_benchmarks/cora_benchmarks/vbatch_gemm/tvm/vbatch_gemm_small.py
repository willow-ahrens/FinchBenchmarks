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

BATCH_SIZE = te.var('bs')
HEAD_SIZE = 512
TILE=64
RTILE=4
# MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)
MAX_LEN = 384

ms = tvm.decl_buffer((BATCH_SIZE,), name = 'ms', dtype = 'int32')
ns = tvm.decl_buffer((BATCH_SIZE,), name = 'ns', dtype = 'int32')
ks = tvm.decl_buffer((BATCH_SIZE,), name = 'ks', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')

def mlbw(name): return Ufw(name, "l", (0, MAX_LEN), [bd], [ms], lambda ms: lambda b: utils.floormult(ms.vload(b), 64))
mlbufw = mlbw('mlb')
def nlbw(name): return Ufw(name, "l", (0, MAX_LEN), [bd], [ns], lambda ns: lambda b: utils.floormult(ns.vload(b), 64))
nlbufw = nlbw('nlb')
if args.split:
    def mubw(name): return Ufw(name, "l", (32, MAX_LEN), [bd], [ms], lambda ms: lambda b: utils.ceilmult(ms.vload(b), 32))
    mubufw = mubw('mub')
    def nubw(name): return Ufw(name, "l", (32, MAX_LEN), [bd], [ns], lambda ns: lambda b: utils.ceilmult(ns.vload(b), 32))
    nubufw = nubw('nub')
else:
    def mubw(name): return Ufw(name, "l", (64, MAX_LEN), [bd], [ms], lambda ms: lambda b: utils.ceilmult(ms.vload(b), 64))
    mubufw = mubw('mub')
    def nubw(name): return Ufw(name, "l", (64, MAX_LEN), [bd], [ns], lambda ns: lambda b: utils.ceilmult(ns.vload(b), 64))
    nubufw = nubw('nub')

def kubw(name): return Ufw(name, "l", (16, MAX_LEN), [bd], [ks], lambda ks: lambda b: utils.ceilmult(ks.vload(b), 16))
kubufw = kubw('kub')

bd_uf = Uf.from_constant('bd', BATCH_SIZE, "l")
mlb_uf = mlbufw.get_uf()
mub_uf = mubufw.get_uf()
nlb_uf = nlbufw.get_uf()
nub_uf = nubufw.get_uf()
kub_uf = kubufw.get_uf()

loop_ufs=[bd_uf, mub_uf, kub_uf]
width_ufs = None # if args.dense_storage else loop_ufs
Q = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, MAX_LEN), [bd, md, kd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[bd_uf, kub_uf, nub_uf]
width_ufs = None # if args.dense_storage else loop_ufs
K = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, MAX_LEN), [bd, kd, nd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[bd_uf, mub_uf, nub_uf]
Op = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, MAX_LEN), [bd, md, nd], loop_ufs, name='Op', width_ufs=None, dtype='float32')

loop_ufs=[bd_uf, mub_uf, nub_uf]
width_ufs = None # if args.dense_storage else [loop_ufs]
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, MAX_LEN), [bd, md, nd], loop_ufs,
                      lambda ds, rds: tvm.sum(Q[ds[bd], ds[md], rds['k']] * K[ds[bd], rds['k'], ds[nd]],
                                              axis = rds['k'], dimensions=[kd]),
                      reduce_axis_ufs = [('k', kub_uf)],
                      name = 'S', width_uf_lists=width_ufs)

alpha = 0.01
beta = 0.03
loop_ufs=[bd_uf, mub_uf, nub_uf]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, MAX_LEN), [bd, md, nd], loop_ufs,
                      lambda ds: alpha*S[ds[bd], ds[md], ds[nd]] + beta*Op[ds[bd], ds[md], ds[nd]],
                      name = 'O', width_uf_lists=None)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")
ntx = 8
nty = 8

def schedule_op(S, O, tile_x, tile_y, suffix):
    if False:
        # S = s.cache_write(O, 'local')
        s[S].set_scope('local')

        Qs = s.cache_read(Q, "shared", [S], layouts='dense', suffix=suffix)
        Ks = s.cache_read(K, "shared", [S], layouts='dense', suffix=suffix)

        Ql = s.cache_read(Qs, "local", [S], layouts='dense', suffix=suffix)
        Kl = s.cache_read(Ks, "local", [S], layouts='dense', suffix=suffix)

        b, x, y = s[O].leaf_iter_vars[0:3]
        xo, xi = s[O].split(x, factor=tile_x)
        yo, yi = s[O].split(y, factor=tile_y)

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
        ko, ki = s[S].split(k, factor=16)
        s[S].reorder(ko, ki, x, y)
        s[Qs].compute_at(s[S], ko)
        s[Ks].compute_at(s[S], ko)
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
        s.reorder_tensor_dimensions(Qs, 1, 2)
    else:
        S_b, S_l, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)

        S_l_o_i, S_l_i = s[S].split(S_l, factor=2)

        S_k_o_o, S_k_o_i = s[S].split(S_k, factor=16)
        s[S].reorder(S_b, S_o, S_k_o_o, S_k_o_i, S_l_o_i, S_l_i)

        O_b, O_l, O_o = tuple(O.op.axis) + tuple(O.op.reduce_axis)

        xo, xi = s[O].split(O_l, factor = tile_x)
        yo, yi = s[O].split(O_o, factor = tile_y)
        s[O].reorder(O_b, xo, yo, xi, yi)
        f1 = s[O].fuse(xo, yo)
        O_b = s[O].fuse(O_b, f1)
        O_l = xi
        O_o = yi

        if tile_x == 64: O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
        else: O_l_o_i, O_l_i = s[O].split(O_l, factor=4)
        O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=4)
        O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

        O_o_o_o_i, O_o_o_i = s[O].split(O_o, factor=32)
        if tile_y == 64: O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=2)
        else: O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
        s[O].reorder(O_b, O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i)

        Q_shared = s.cache_read(Q, "shared", [S], suffix=suffix)
        Q_shared_ax0, Q_shared_ax1, Q_shared_ax2 = tuple(Q_shared.op.axis)
        s[Q_shared].compute_at(s[S], S_k_o_o)

        K_shared = s.cache_read(K, "shared", [S], suffix=suffix)
        K_shared_ax0, K_shared_ax1, K_shared_ax2 = tuple(K_shared.op.axis)
        s[K_shared].compute_at(s[S], S_k_o_o)

        O_b_l_o_o_o_fused_o_o_o_o_fused = s[O].fuse(O_b, O_l_o_o_o, O_o_o_o_o)
        s[O].bind(O_b_l_o_o_o_fused_o_o_o_o_fused, te.thread_axis("blockIdx.x"))
        O_l_o_o_i_fused_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
        s[O].bind(O_l_o_o_i_fused_o_o_o_i_fused, te.thread_axis("vthread"), no_unroll_vthread=True)
        O_l_o_i_fused_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
        s[O].bind(O_l_o_i_fused_o_o_i_fused, te.thread_axis("threadIdx.x"))
        s[S].compute_at(s[O], O_l_o_i_fused_o_o_i_fused)

        Q_shared_ax1_f_ax2_f = s[Q_shared].fuse(Q_shared_ax1, Q_shared_ax2)
        Q_shared_ax1_f_ax2_f_o, Q_shared_ax1_f_ax2_f_i = s[Q_shared].split(Q_shared_ax1_f_ax2_f, factor=4)
        s[Q_shared].vectorize(Q_shared_ax1_f_ax2_f_i)
        Q_shared_ax1_f_ax2_f_o_o, Q_shared_ax1_f_ax2_f_o_i = s[Q_shared].split(Q_shared_ax1_f_ax2_f_o, factor=128)
        s[Q_shared].bind(Q_shared_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

        K_shared_ax1_f_ax2_f = s[K_shared].fuse(K_shared_ax1, K_shared_ax2)
        K_shared_ax1_f_ax2_f_o, K_shared_ax1_f_ax2_f_i = s[K_shared].split(K_shared_ax1_f_ax2_f, factor=4)
        s[K_shared].vectorize(K_shared_ax1_f_ax2_f_i)
        K_shared_ax1_f_ax2_f_o_o, K_shared_ax1_f_ax2_f_o_i = s[K_shared].split(K_shared_ax1_f_ax2_f_o, factor=128)
        s[K_shared].bind(K_shared_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

        s[S].set_scope('local')

if args.split:
    G1, G2, G3, G4 = s.split_for_bin_packing([S], O, {O.op.axis[1]: mlb_uf, O.op.axis[2]: nlb_uf}, include_inputs=True)
    S1, O1, = G1
    S2, O2, = G2
    S3, O3, = G3
    S4, O4, = G4
    schedule_op(S1, O1, 64, 64, '1')
    schedule_op(S2, O2, 32, 64, '2')
    schedule_op(S3, O3, 64, 32, '3')
    schedule_op(S4, O4, 32, 32, '4')

    if args.hfuse:
        s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0]),
                 (s[O3].op, s[O3].leaf_iter_vars[0]), (s[O4].op, s[O4].leaf_iter_vars[0])])

    # G1, G2 = s.split_for_bin_packing([S], O, {O.op.axis[1]: mlb_uf}, include_inputs=True)
    # S1, O1, = G1
    # S2, O2, = G2
    # schedule_op(S1, O1, 64, 64, '1')
    # schedule_op(S2, O2, 32, 64, '2')

    # if args.hfuse:
        # s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0])])
else:
    schedule_op(S, O, 64, 64, '1')


gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    ms = l_inputs[0]
    return {
        # Q: 3 * NUM_HEADS * MAX_LEN * run_utils.prefix_sum(len(ms), lambda b: (mubufw.get_fn(ms)(b))),
        # K: 3 * NUM_HEADS * MAX_LEN * run_utils.prefix_sum(len(ms), lambda b: (mubufw.get_fn(ms)(b))),
        # O: NUM_HEADS * run_utils.prefix_sum(len(ms),
                                            # lambda b: (mubufw.get_fn(ms)(b) *
                                                       # mubufw.get_fn(ms)(b)))
    }

bO = tvm.tir.decl_buffer((BATCH_SIZE, MAX_LEN, MAX_LEN), name="bO")
inputs = [[ms, ns, ks], [BATCH_SIZE, Q, K, bO]]
if args.split:
    binds = {O1:bO, O2:bO, O3:bO, O4:bO, Op:bO}
    # binds = {O1:bO, O2:bO, Op:bO}
else:
    binds = {O:bO, Op:bO}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds,
                               run_function=run_utils.get_vbatch_gemm_run_fn(BATCH_SIZE, no_scale=True))
# _, Q, K, O = out
# for i in range(BATCH_SIZE):
#     length = batches[0][i]
#     rounded = utils.ceilmult(length, TILE)
#     print(rounded, np.mean(O[i,0:rounded,:,0:rounded]))
