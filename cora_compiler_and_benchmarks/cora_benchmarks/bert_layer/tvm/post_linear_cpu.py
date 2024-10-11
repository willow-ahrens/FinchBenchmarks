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
args = parser.parse_args()

args.target = run_utils.get_arm_target()

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

# W = te.placeholder((NUM_HEADS * HEAD_SIZE, OUT_SIZE), name='W')
loop_ufs=[ls[4], ls[1], ls[3]]
width_ufs=[ls[4], ls[1], ls[3]]
W = te.ragged_placeholder((OUT_SIZE, NUM_HEADS, HEAD_SIZE), [od, md, hd], loop_ufs,
                          name='W', width_ufs=width_ufs)

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
                                         W[ds[od], tvm.floordiv(k, HEAD_SIZE), tvm.floormod(k, HEAD_SIZE)],
                                         axis=k, dimensions = [mdhd]),
                      name = 'S', width_uf_lists=width_ufs)

def compute_body(ds):
    if args.skip_residual: return S[ds[bd], ds[s1], ds[od]] + B[ds[od]]
    else: return A2[ds[bd], ds[s1], ds[od]] + S[ds[bd], ds[s1], ds[od]] + B[ds[od]]
loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=None if args.dense_storage else [loop_ufs]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      compute_body, name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if False:
    O_local = S

    As = s.cache_read(A, "shared", [S], loop_layout=[ls[0], ls[2], ls[1], ls[3]], layouts=[ls[0], ls[2], ls[1], ls[3]])
    # As = s.cache_read(A, "shared", [O_local], layouts='dense')
    Ws = s.cache_read(W, "shared", [O_local], vanilla=True)

    O_local_b_c, O_local_m_c, O_local_n_c, O_local_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    O_local_m_c = s[O_local].fuse(O_local_b_c, O_local_m_c, padding=4)
    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=4)
    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=64)
    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=16)
    s[O_local].reorder(O_local_m_c_o_i, O_local_n_c_o_i, O_local_k_o, O_local_k_i, O_local_m_c_i, O_local_n_c_i)

    b, x, y = s[O].leaf_iter_vars[0:3]
    x = s[O].fuse(b, x, padding=64)
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(xo, yo, xi, yi)
    f = s[O].fuse(xo, yo)
    s[O].parallel(f)

    O_m, O_n = xi, yi
    O_m_o_i, O_m_i = s[O].split(O_m, factor=4)

    O_n_o_i, O_n_i = s[O].split(O_n, factor=64)
    O_n_o_o, O_n_o_i = s[O].split(O_n_o_i, factor=1)
    s[O].reorder(O_m_o_i, O_n_o_o, O_n_o_i, O_m_i, O_n_i)
    s[O_local].compute_at(s[O], O_n_o_i)
    s[As].compute_at(s[O], O_n_o_i)
    s[Ws].compute_at(s[O_local], O_local_k_o)

    s[O_local].vectorize(O_local_n_c_i)
    s[O].vectorize(O_n_i)

    inputs = [[lens], [BS_VAR, A, W, O]]

    s.fuse_tensor_dimensions(O_local, 0, 1)
    s.fuse_tensor_dimensions(As, 0, 1)

    b, l, h, i = s[As].leaf_iter_vars
    s[As].fuse(b, l)

    s[S].set_scope('local')
    s[S].mark_no_bounds_check()
    s[As].mark_no_bounds_check()
else:
    O_local = S

    Wl = s.cache_read(W, 'local', [O_local])

    O_local_b_c, O_local_m_c, O_local_n_c, O_local_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    O_local_m_c = s[O_local].fuse(O_local_b_c, O_local_m_c)

    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=4)
    O_local_m_c_o_o_i, O_local_m_c_o_i = s[O_local].split(O_local_m_c_o_i, factor=16)

    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=4)
    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=64)

    s[O_local].reorder(O_local_m_c_o_o_i, O_local_k_o, O_local_m_c_o_i, O_local_n_c_o_i, O_local_k_i, O_local_m_c_i, O_local_n_c_i)
    s[Wl].compute_at(s[O_local], O_local_k_o)

    s[O_local].unroll(O_local_n_c_i)
    s[O_local].unroll(O_local_m_c_i)

    b, x, y = s[O].leaf_iter_vars[0:3]
    x = s[O].fuse(b, x, padding=64)
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(xo, yo, xi, yi)
    f = s[O].fuse(xo, yo)
    s[O].parallel(f)

    O_m, O_n = xi, yi
    O_m_o_i, O_m_i = s[O].split(O_m, factor=64)
    O_m_o_o, O_m_o_i = s[O].split(O_m_o_i, factor=1)
    O_n_o_i, O_n_i = s[O].split(O_n, factor=16)
    O_n_o_o, O_n_o_i = s[O].split(O_n_o_i, factor=4)
    s[O].reorder(O_m_o_o, O_n_o_o, O_m_o_i, O_n_o_i, O_m_i, O_n_i)
    s[O_local].compute_at(s[O], O_n_o_i)

    s[O_local].vectorize(O_local_n_c_i)
    s[O].vectorize(O_n_i)

    s[O_local].set_scope('local')

    s.fuse_tensor_dimensions(O_local, 0, 1)
    s[S].mark_no_bounds_check()

    s.split_tensor_dimension(Wl, 0, 4)
    s.reorder_tensor_dimensions(Wl, 1, 2)
    s.reorder_tensor_dimensions(Wl, 2, 3)

def size_fn(l_inputs):
    lens = l_inputs[0]

    return {
        A: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(
            len(lens), lambda b: (lufw1 if args.layout_unfused else lufw64).get_fn(lens)(b)),
        A2: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                        run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b))),
        O: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                       run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)))
    }

if args.skip_residual:
    inputs = [[lens], [BS_VAR, A, W, B, O]]
else:
    inputs = [[lens], [BS_VAR, A, A2, W, B, O]]
binds = {}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

# _, A, W, B, O = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]))
#     ctr += this_extent
