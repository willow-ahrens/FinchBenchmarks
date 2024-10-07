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

parser = run_utils.get_cmd_parser()
parser.add_argument('--hfuse', dest='hfuse', default=False, action='store_true')
args = parser.parse_args()

BATCH_SIZE = te.var('bs')
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
RTILE=4
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

assert MAX_LEN > 64

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')
qk = Dim('qk')

def lbw(name): return Ufw(name, "l", (0, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.floormult(lens[b], 64))
def ubw(name): return Ufw(name, "l", (32, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], 32))
lbufw = lbw('lb')
ubufw = ubw('ub')

qk_uf = Uf.from_constant('qk', 3, "l")
bd_uf = Uf.from_constant('bd', BATCH_SIZE, "l")
md_uf = Uf.from_constant('md', NUM_HEADS, "l")
lb_uf = lbufw.get_uf()
ub_uf = ubufw.get_uf()
hd_uf = Uf.from_constant('hd', HEAD_SIZE, "l")

loop_ufs=[qk_uf, bd_uf, ub_uf, md_uf, hd_uf]
width_ufs = None if args.dense_storage else loop_ufs
Q = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[qk_uf, bd_uf, ub_uf, md_uf, hd_uf]
width_ufs = None if args.dense_storage else loop_ufs
K = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[bd_uf, ub_uf, md_uf, ub_uf]
width_ufs = None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[0, ds[bd], ds[s1], ds[md], k] * K[1, ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions=[hd]),
                      name = 'S', width_uf_lists=width_ufs)

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.if_then_else(ds[s1] >= lens[ds[bd]], -float('inf'), S[ds[bd], ds[s1], ds[md], ds[s2]]),
                      name = 'O', width_uf_lists=width_ufs)

output_layout = O.op.output_layout(0)
s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")
ntx = 8
nty = 8

def schedule_op(S, O, tile_x, tile_y, suffix):
    S_b, S_l, S_h, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)

    S_l_o_i, S_l_i = s[S].split(S_l, factor=2)

    S_k_o_o, S_k_o_i = s[S].split(S_k, factor=16)
    s[S].reorder(S_b, S_o, S_k_o_o, S_k_o_i, S_l_o_i, S_l_i)

    O_b, O_l, O_h, O_o = tuple(O.op.axis) + tuple(O.op.reduce_axis)

    xo, xi = s[O].split(O_l, factor = tile_x)
    yo, yi = s[O].split(O_o, factor = tile_y)
    s[O].reorder(O_h, O_b, xo, yo, xi, yi)
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
    s[O].reorder(O_b, O_l_o_o_o, O_o_o_o_o, O_h, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i)

    Q_shared = s.cache_read(Q, "shared", [S], suffix=suffix)
    Q_shared_axm2, Q_shared_axm1, Q_shared_ax0, Q_shared_ax1, Q_shared_ax2 = tuple(Q_shared.op.axis)
    s[Q_shared].compute_at(s[S], S_k_o_o)

    K_shared = s.cache_read(K, "shared", [S], suffix=suffix)
    K_shared_axm2, K_shared_axm1, K_shared_ax0, K_shared_ax1, K_shared_ax2 = tuple(K_shared.op.axis)
    s[K_shared].compute_at(s[S], S_k_o_o)

    O_b_l_o_o_o_fused_o_o_o_o_fused = s[O].fuse(O_b, O_l_o_o_o, O_o_o_o_o)
    s[O].bind(O_b_l_o_o_o_fused_o_o_o_o_fused, te.thread_axis("blockIdx.x"))
    s[O].bind(O_h, te.thread_axis("blockIdx.y"))
    O_l_o_o_i_fused_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
    s[O].bind(O_l_o_o_i_fused_o_o_o_i_fused, te.thread_axis("vthread"))
    O_l_o_i_fused_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
    s[O].bind(O_l_o_i_fused_o_o_i_fused, te.thread_axis("threadIdx.x"))
    s[S].compute_at(s[O], O_l_o_i_fused_o_o_i_fused)

    Q_shared_ax0_ax1_f_ax2_f = s[Q_shared].fuse(Q_shared_ax0, Q_shared_ax1, Q_shared_ax2)
    Q_shared_ax0_ax1_f_ax2_f_o, Q_shared_ax0_ax1_f_ax2_f_i = s[Q_shared].split(Q_shared_ax0_ax1_f_ax2_f, factor=4)
    s[Q_shared].vectorize(Q_shared_ax0_ax1_f_ax2_f_i)
    Q_shared_ax0_ax1_f_ax2_f_o_o, Q_shared_ax0_ax1_f_ax2_f_o_i = s[Q_shared].split(Q_shared_ax0_ax1_f_ax2_f_o, factor=128)
    s[Q_shared].bind(Q_shared_ax0_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

    K_shared_ax0_ax1_f_ax2_f = s[K_shared].fuse(K_shared_ax0, K_shared_ax1, K_shared_ax2)
    K_shared_ax0_ax1_f_ax2_f_o, K_shared_ax0_ax1_f_ax2_f_i = s[K_shared].split(K_shared_ax0_ax1_f_ax2_f, factor=4)
    s[K_shared].vectorize(K_shared_ax0_ax1_f_ax2_f_i)
    K_shared_ax0_ax1_f_ax2_f_o_o, K_shared_ax0_ax1_f_ax2_f_o_i = s[K_shared].split(K_shared_ax0_ax1_f_ax2_f_o, factor=128)
    s[K_shared].bind(K_shared_ax0_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

    s[S].set_scope('local')

G1, G2, G3, G4 = s.split_for_bin_packing([S], O, {O.op.axis[1]: lb_uf, O.op.axis[3]: lb_uf}, include_inputs=True)
S1, O1 = G1
S2, O2 = G2
S3, O3 = G3
S4, O4 = G4
if args.target == "cuda":
    schedule_op(S1, O1, 64, 64, '1')
    schedule_op(S2, O2, 32, 64, '2')
    schedule_op(S3, O3, 64, 32, '3')
    schedule_op(S4, O4, 32, 32, '4')

if args.hfuse:
    s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0]),
             (s[O3].op, s[O3].leaf_iter_vars[0]), (s[O4].op, s[O4].leaf_iter_vars[0])])

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        Q: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (ubufw.get_fn(lens)(b))),
        K: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (ubufw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (ubufw.get_fn(lens)(b) *
                                                       ubufw.get_fn(lens)(b)))
    }

bO = tvm.tir.decl_buffer(output_layout, name="bO")
if args.target == "cuda":
    inputs = [[lens], [BATCH_SIZE, Q, K, bO]]
else:
    inputs = [[lens], [BATCH_SIZE, Q, K, S1, S2, S3, S4, bO]]
binds = {O1:bO, O2:bO, O3:bO, O4:bO}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, hoist_loads=True,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE))
# _, Q, K, O = out
# for i in range(BATCH_SIZE):
#     length = batches[0][i]
#     rounded = utils.ceilmult(length, TILE)
#     print(rounded, np.mean(O[i,0:rounded,:,0:rounded]))

O = out[-1]
O = O.flatten()
ctr = 0
for length in batches[0]:
    rounded = utils.ceilmult(length, 32)
    this_extent = rounded
    this_storage_extent = rounded * rounded * NUM_HEADS
    print(rounded, np.mean(O[ctr:ctr+length]))
    ctr += this_storage_extent
