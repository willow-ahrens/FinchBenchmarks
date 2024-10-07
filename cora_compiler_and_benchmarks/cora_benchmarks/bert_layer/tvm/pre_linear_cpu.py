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
NUM_HEADS = 8
IN_SIZE = 512
OUT_SIZE = 64
QKV_NUM = 3
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
qkv = Dim('qkv')
md = Dim('md')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('qkv', QKV_NUM, "l"),
    1: Uf.from_constant('bd', BATCH_SIZE, "l"),
    2: Uf.from_constant('md', NUM_HEADS, "l"),
    3: lufw1.get_uf(),
    4: Uf.from_constant('id', IN_SIZE, "l"),
    5: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[1], ls[3], ls[4]]
width_ufs=[ls[1], ls[3], ls[4]]
QKV = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                            name='QKV', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[2], ls[5], ls[4]]
width_ufs=[ls[0], ls[2], ls[5], ls[4]]
W = te.ragged_placeholder((QKV_NUM, NUM_HEADS, OUT_SIZE, IN_SIZE), [qkv, md, od, id], loop_ufs,
                          name='W', width_ufs=width_ufs)

# W = te.placeholder((QKV_NUM, NUM_HEADS, OUT_SIZE, IN_SIZE), name='W')
B = te.placeholder((QKV_NUM, NUM_HEADS, OUT_SIZE), name='B')

loop_ufs=[ls[0], ls[1], ls[3], ls[2], ls[5]]
width_ufs=None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
S = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od], loop_ufs,
                      lambda ds: tvm.sum(W[ds[qkv], ds[md], ds[od], k] * QKV[ds[bd], ds[s1], k],
                                         axis = k, dimensions = [id]),
                      name = 'S', width_uf_lists=width_ufs)

if args.layout_unfused:
    width_ufs=None if args.dense_storage else [[ls[0], ls[1], lufw1.get_uf(), ls[2], ls[5]]]
else:
    width_ufs=None if args.dense_storage else [[ls[0], ls[1], lufw64.get_uf(), ls[2], ls[5]]]
O = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od], loop_ufs,
                      lambda ds: S[ds[qkv], ds[bd], ds[s1], ds[md], ds[od]] + B[ds[qkv], ds[md], ds[od]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if True:
    O_local = S

    Wl = s.cache_read(W, 'local', [O_local], layouts='dense')
    # QKVl = s.cache_read(QKV, 'local', [O_local])

    O_local_q_c, O_local_b_c, O_local_m_c, O_local_h_c, O_local_n_c, O_local_k = (tuple(O_local.op.axis) +
                                                                                  tuple(O_local.op.reduce_axis))
    O_local_m_c = s[O_local].fuse(O_local_b_c, O_local_m_c)

    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=4)
    O_local_m_c_o_o_i, O_local_m_c_o_i = s[O_local].split(O_local_m_c_o_i, factor=16)

    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=4)
    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=64)

    s[O_local].reorder(O_local_m_c_o_o_i, O_local_k_o, O_local_m_c_o_i, O_local_n_c_o_i, O_local_k_i, O_local_m_c_i, O_local_n_c_i)
    s[Wl].compute_at(s[O_local], O_local_k_o)

    s[O_local].unroll(O_local_m_c_i)

    q, b, x, h, y = s[O].leaf_iter_vars
    x = s[O].fuse(b, x, padding=64)
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(xo, yo, q, h, xi, yi)
    f = s[O].fuse(xo, yo)
    f = s[O].fuse(f, q, h)
    s[O].parallel(f)

    O_m, O_n = xi, yi
    O_m_o_i, O_m_i = s[O].split(O_m, factor=64)
    O_m_o_o, O_m_o_i = s[O].split(O_m_o_i, factor=1)
    O_n_o_i, O_n_i = s[O].split(O_n, factor=16)
    O_n_o_o, O_n_o_i = s[O].split(O_n_o_i, factor=4)
    s[O].reorder(O_m_o_o, O_n_o_o, O_m_o_i, O_n_o_i, O_m_i, O_n_i)
    s[O_local].compute_at(s[O], O_n_o_i)

    # s[QKVl].compute_at(s[O], O_n_o_i)
    # b, l = s[QKVl].leaf_iter_vars[0:2]
    # s[QKVl].fuse(b, l)
    # s.fuse_tensor_dimensions(QKVl, 0, 1)
    # s[QKVl].mark_no_bounds_check()

    s[O_local].vectorize(O_local_n_c_i)
    s[O].vectorize(O_n_i)

    s[O_local].set_scope('local')

    s.fuse_tensor_dimensions(O_local, 1, 2)
    s[S].mark_no_bounds_check()

    s.split_tensor_dimension(Wl, 2, 4)
    s.reorder_tensor_dimensions(Wl, 3, 4)

else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    if args.layout_unfused: out_fn = lufw1.get_fn(lens)
    else: out_fn = lufw64.get_fn(lens)

    return {
        QKV: IN_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)),
        O: QKV_NUM * NUM_HEADS * OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                                             run_utils.prefix_sum(len(lens), lambda b: out_fn(b)))
    }

# bQKV = tvm.decl_buffer([BATCH_SIZE*MAX_LEN, IN_SIZE], name = "bQKV")
# binds = {QKV: bQKV}
binds={}
inputs = [[lens], [BS_VAR, QKV, W, B, O]]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

q_size = 0
for length in batches[0]:
    q_size += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE

# ctr = 0
# _, QKV, W, B, O  = out
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * NUM_HEADS * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]), np.mean(O[ctr+q_size:ctr+q_size + this_extent]),
#           np.mean(O[ctr+2*q_size:ctr+2*q_size + this_extent]))
#     ctr += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE
