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
parser.add_argument('--kt', dest='kt', default=8, type=int)
# parser.add_argument('--nt', dest='nt', default=16, type=int)
args = parser.parse_args()

args.target = run_utils.get_arm_target()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
HEAD_SIZE = 64

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

if args.no_raggedness:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [], [], lambda : lambda : utils.ceilmult(MAX_LEN, pad))
else:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s', 16)

if args.dataset in ['mprc', 'cola']: lufwp = len_ufw('s', 32)
else: lufwp = len_ufw('s', 64)
sufwp = len_ufw('s', 64)

lbduf = Uf.from_constant('bd', BS_VAR, "l")
ls = {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufwp.get_uf(),
    3: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    4: Uf.from_constant('qk', 3, "l"),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[2]]
width_ufs=[ls[0], sufwp.get_uf(), ls[1], sufwp.get_uf()]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s2, md, s1], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[4], ls[0], ls[2], ls[1], ls[3]]
width_ufs=[ls[4], ls[0], sufwp.get_uf(), ls[1], ls[3]]
V = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[lbduf, ls[2], ls[1], ls[3]]
width_ufs=None if args.dense_storage else [[ls[0], sufwp.get_uf(), ls[1], ls[3]]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s2, md, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[s2], ds[md], rds['k']] *
                                              V[2, ds[bd], rds['k'], ds[md], ds[hd]],
                                              axis=rds['k'], dimensions=[s1]),
                      name = 'O', reduce_axis_ufs = [('k', lufw1.get_uf())],
                      width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if False:
    O_local = s.cache_write(O, 'local')

    As = s.cache_read(A, "shared", [O_local], layouts='dense')
    Vs = s.cache_read(V, "shared", [O_local], layouts='dense')

    O_local_b_c, O_local_m_c, O_local_h_c, O_local_n_c, O_local_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=4)
    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=64)
    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=16)
    s[O_local].reorder(O_local_b_c, O_local_h_c, O_local_m_c_o_i, O_local_n_c_o_i, O_local_k_o, O_local_k_i, O_local_m_c_i, O_local_n_c_i)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(b, xo, yo, h, xi, yi)
    f1 = s[O].fuse(xo, yo)
    f2 = s[O].fuse(b, f1)
    s[O].parallel(f2)

    O_m, O_n = xi, yi
    O_m_o_i, O_m_i = s[O].split(O_m, factor=4)

    O_n_o_i, O_n_i = s[O].split(O_n, factor=64)
    O_n_o_o, O_n_o_i = s[O].split(O_n_o_i, factor=1)
    s[O].reorder(O_m_o_i, O_n_o_o, O_n_o_i, O_m_i, O_n_i)
    s[O_local].compute_at(s[O], O_n_o_i)
    s[As].compute_at(s[O], O_n_o_i)
    s[Vs].compute_at(s[O_local], O_local_k_o)

    s[O_local].vectorize(O_local_n_c_i)
    s[O].vectorize(O_n_i)

    inputs = [[lens], [BS_VAR, A, V, O]]
else:
    O_local, = s.cache_write([O], "local")

    Al = s.cache_read(A, "local", [O_local], layouts='dense')

    O_local_b_c, O_local_m_c, O_local_h_c, O_local_n_c, O_local_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=4)
    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=4)
    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=16)

    s[O_local].reorder(O_local_b_c, O_local_k_o, O_local_m_c_o_i, O_local_n_c_o_i, O_local_k_i, O_local_m_c_i, O_local_n_c_i)
    s[Al].compute_at(s[O_local], O_local_k_o)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = 64)
    yo, yi = s[O].split(y, factor = 64)
    s[O].reorder(b, xo, yo, h, xi, yi)
    f1 = s[O].fuse(xo, yo)
    f2 = s[O].fuse(b, f1)
    s[O].parallel(f2)

    s[O_local].unroll(O_local_m_c_i)
    s[O_local].unroll(O_local_n_c_i)
    # s[O_local].peel(O_local_k_o)

    O_m, O_n = xi, yi
    O_m_o_i, O_m_i = s[O].split(O_m, factor=16)
    O_n_o_i, O_n_i = s[O].split(O_n, factor=64)
    s[O].reorder(O_m_o_i, O_n_o_i, O_m_i, O_n_i)
    s[O_local].compute_at(s[O], O_n_o_i)

    s[O_local].vectorize(O_local_n_c_i)
    s[O].vectorize(O_n_i)

    s.split_tensor_dimension(Al, 1, 4)
    s.reorder_tensor_dimensions(Al, 2, 3)
    s.reorder_tensor_dimensions(Al, 3, 4)

def size_fn(l_inputs):
    if args.no_raggedness: return {}
    else:
        lens = l_inputs[0]
        return {
            V: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (sufwp.get_fn(lens)(b))),
            A: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (sufwp.get_fn(lens)(b) *
                                                                      sufwp.get_fn(lens)(b))),
            O: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: sufwp.get_fn(lens)(b))
        }

prep_code_mode = 'no_prep_code' if args.no_raggedness else 'with_prep_code'
inputs = [[lens], [BS_VAR, V, A, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR),
                                        prep_code_mode=prep_code_mode)

# _, V, A, O  = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     rounded64 = utils.ceilmult(length, 64)
#     this_extent = rounded64 * NUM_HEADS * HEAD_SIZE
#     print(length, run_utils.stats(O[ctr:ctr + this_extent]))
#     ctr += this_extent
