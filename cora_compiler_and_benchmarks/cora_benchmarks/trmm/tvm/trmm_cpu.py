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
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--m', dest='m', default=1024, type=int)
parser.add_argument('--n', dest='n', default=128, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--load-balance', dest='load_balance', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--op-split', dest='op_split', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')

parser.add_argument('--m1', dest='m1', default=2, type=int)
parser.add_argument('--m2', dest='m2', default=1, type=int)
parser.add_argument('--n1', dest='n1', default=32, type=int)
parser.add_argument('--n2', dest='n2', default=4, type=int)
parser.add_argument('--k1', dest='k1', default=8, type=int)
parser.add_argument('--k2', dest='k2', default=8, type=int)
args = parser.parse_args()

M = args.m
N = args.n
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')

ls =  {
    0: Uf.from_constant('md', M, 'l'),
    1: Uf.from_constant('nd', N, 'l'),
    2: Uf.from_constant('kd', M, 'l'),
}

loop_ufs=[ls[0], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((M, M), [md, kd], loop_ufs, name='A', width_ufs=None)

B = te.placeholder((M, N), name='B')

alpha = 2
if args.op_split:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.floormult(m, pad))
    luf = len_ufw('s2k', 128).get_uf()

    loop_ufs=[ls[0], ls[1]]
    O1 = te.ragged_compute((M, N), [md, nd], loop_ufs,
                           lambda ds, rds: tvm.sum(A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                                   axis=rds['k'], dimensions = [kd]),
                           name = 'O1', reduce_axis_ufs = [('k', luf)], width_uf_lists=None)

    O2i = te.ragged_compute((M, N), [md, nd], loop_ufs,
                            lambda ds, rds: tvm.sum(tvm.tir.Cast('int32', utils.floormult(ds[md], 32) + rds['k'] < (ds[md] + 1)) *
                                                    A[ds[md], utils.floormult(ds[md], 32) + rds['k']] *
                                                    B[utils.floormult(ds[md], 32) + rds['k'], ds[nd]],
                                                    axis=rds['k'], dimensions = [kd]),
                            name = 'O2i', reduce_axis_ufs = [('k', Uf.from_constant('kd', 32, 'l'))], width_uf_lists=None)

    O2 = te.ragged_compute((M, N), [md, nd], loop_ufs,
                           lambda ds: alpha*(O1[ds[md], ds[nd]] + O2i[ds[md], ds[nd]]),
                           name = 'O2', width_uf_lists=None)

    s = tvm.create_schedule([O1.op, O2.op])
else:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.ceilmult(m + 1, pad))
    luf = len_ufw('s2k', 256).get_uf()

    loop_ufs=[ls[0], ls[1]]
    S = te.ragged_compute((M, N), [md, nd], loop_ufs,
                          lambda ds, rds: tvm.sum(tvm.tir.Cast('int32', rds['k'] < (ds[md] + 1)) *
                                                  A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                                  axis=rds['k'], dimensions = [kd]),
                          # lambda ds, rds: tvm.sum(A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                                  # axis=rds['k'], dimensions = [kd]),
                          name = 'S', reduce_axis_ufs = [('k', luf)], width_uf_lists=None)

    O = te.ragged_compute((M, N), [md, nd], loop_ufs, lambda ds: alpha*S[ds[md], ds[nd]], name = 'O', width_uf_lists=None)

    s = tvm.create_schedule([O.op])

def schedule_op(O, suffix, cache_write_tensor=None):
    if cache_write_tensor is not None:
        O_local = cache_write_tensor
    else:
        O_local, = s.cache_write([O], "local", storage_layout_mode='loop_layout')

    O_local_m_c, O_local_n_c, O_local_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    O_local_m_c_o_i, O_local_m_c_i = s[O_local].split(O_local_m_c, factor=4)
    O_local_m_c_o_o_i, O_local_m_c_o_i = s[O_local].split(O_local_m_c_o_i, factor=64)
    O_local_m_c_o_o_o, O_local_m_c_o_o_i = s[O_local].split(O_local_m_c_o_o_i, factor=1)

    O_local_n_c_o_i, O_local_n_c_i = s[O_local].split(O_local_n_c, factor=64)
    O_local_n_c_o_o_i, O_local_n_c_o_i = s[O_local].split(O_local_n_c_o_i, factor=1)
    O_local_n_c_o_o_o, O_local_n_c_o_o_i = s[O_local].split(O_local_n_c_o_o_i, factor=1)

    O_local_k_o, O_local_k_i = s[O_local].split(O_local_k, factor=32)
    s[O_local].reorder(O_local_m_c_o_o_o, O_local_n_c_o_o_o, O_local_m_c_o_o_i, O_local_n_c_o_o_i, O_local_k_o, O_local_m_c_o_i, O_local_n_c_o_i, O_local_k_i, O_local_m_c_i, O_local_n_c_i)

    O_m, O_n, = tuple(O.op.axis)
    O_m_o, O_m_i = s[O].split(O_m, factor=256)
    O_n_o, O_n_i = s[O].split(O_n, factor=64)
    s[O].reorder(O_m_o, O_n_o, O_m_i, O_n_i)

    O_m_o_n_o_fused = s[O].fuse(O_m_o, O_n_o)
    s[O].parallel(O_m_o_n_o_fused)
    s[O_local].compute_at(s[O], O_m_o_n_o_fused)
    s[O_local].pragma(O_local_m_c_o_o_o, "auto_unroll_max_step", 512)
    s[O_local].pragma(O_local_m_c_o_o_o, "unroll_explicit", True)
    s[O_local].vectorize(O_local_n_c_i)
    if cache_write_tensor is None: return [O.op, O_local.op]
    else: return []

substitute_ops = []
if args.op_split:
    substitute_ops += schedule_op(O1, '1')
    substitute_ops += schedule_op(O2, '2', O2i)
else:
    substitute_ops += schedule_op(O, '', S)

if args.op_split: inputs = [[], [A, B, O1, O2]]
else: inputs = [[], [A, B, O]]

substitutes=None
if args.load_balance:
    print('Load balancing')
    max_by = (M // 256) * (M // 64)
    substitutes=[substitute_ops, {'iO1_0.o1.o_f': Uf('sub', "", (0, max_by), [Dim('dum')], lambda b: max_by - b - 1)}]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, run_function=run_utils.run_trmm,
                               prep_code_mode='no_prep_code', substitutes=substitutes)

# if args.op_split:
#     A, B, O1, O2  = out
#     for i in range(args.m):
#         print(i + 1, np.mean(O1[i, 0:(i+1)]), np.mean(O2[i, 0:(i+1)]))
# else:
#     A, B, O  = out
#     for i in range(args.m):
#         print(i + 1, np.mean(O[i, 0:(i+1)]))
