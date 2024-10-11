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

parser.add_argument('--m1', dest='m1', default=4, type=int)
parser.add_argument('--m2', dest='m2', default=64, type=int)
parser.add_argument('--n1', dest='n1', default=64, type=int)
parser.add_argument('--n2', dest='n2', default=1, type=int)
parser.add_argument('--k1', dest='k1', default=8, type=int)
parser.add_argument('--k2', dest='k2', default=8, type=int)
args = parser.parse_args()

if args.m1 * args.m2 > 256: exit(0)
if args.n1 * args.n2 > 64: exit(0)

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
loop_ufs=[ls[2], ls[1]]
width_ufs=loop_ufs
B = te.ragged_placeholder((M, M), [kd, nd], loop_ufs, name='B', width_ufs=None)

# factor = args.m1 * args.m2
factor = 256
alpha = 2
if args.op_split:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.floormult(m, pad))
    luf = len_ufw('s2k', factor).get_uf()

    loop_ufs=[ls[0], ls[1]]
    O1 = te.ragged_compute((M, N), [md, nd], loop_ufs,
                           lambda ds, rds: tvm.sum(A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                                   axis=rds['k'], dimensions = [kd]),
                           name = 'O1', reduce_axis_ufs = [('k', luf)], width_uf_lists=None)

    O2i = te.ragged_compute((M, N), [md, nd], loop_ufs,
                            lambda ds, rds: tvm.sum(tvm.tir.Cast('int32', utils.floormult(ds[md], 32) + rds['k'] < (ds[md] + 1)) *
                                                    A[ds[md], utils.floormult(ds[md], 32) + rds['k']] *
                                                    B[utils.floormult(ds[md], 32) + rds['k'], ds[nd]],
                                                    axis=rds['k'], dimensions = [kd]), name = 'O2i',
                            reduce_axis_ufs = [('k', Uf.from_constant('kd', 32, 'l'))], width_uf_lists=None)

    O2 = te.ragged_compute((M, N), [md, nd], loop_ufs,
                           lambda ds: alpha*(O1[ds[md], ds[nd]] + O2i[ds[md], ds[nd]]),
                           name = 'O2', width_uf_lists=None)

    s = tvm.create_schedule([O1.op, O2.op])
else:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.ceilmult(m + 1, pad))
    luf = len_ufw('s2k', factor).get_uf()

    loop_ufs=[ls[0], ls[1]]
    S = te.ragged_compute((M, N), [md, nd], loop_ufs,
                          lambda ds, rds: tvm.sum(tvm.tir.Cast('int32', rds['k'] < (ds[md] + 1)) *
                                                  A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                                  axis=rds['k'], dimensions = [kd]),
                          # lambda ds, rds: tvm.sum(A[ds[md], rds['k']] * B[rds['k'], ds[nd]],
                                                  # axis=rds['k'], dimensions = [kd]),
                          name = 'S', reduce_axis_ufs = [('k', luf)], width_uf_lists=None)

    O = te.ragged_compute((M, N), [md, nd], loop_ufs, lambda ds: alpha*S[ds[md], ds[nd]],
                          name = 'O', width_uf_lists=None)

    s = tvm.create_schedule([O.op])

def schedule_op(O, suffix, cache_write_tensor=None):
    residual = (cache_write_tensor is not None) and args.op_split
    if cache_write_tensor is not None:
        O_l = cache_write_tensor
    else:
        O_l, = s.cache_write([O], "local", storage_layout_mode='loop_layout')

    O_l_m_c, O_l_n_c, O_l_k = tuple(O_l.op.axis) + tuple(O_l.op.reduce_axis)

    O_l_m_c_o_i, O_l_m_c_i = s[O_l].split(O_l_m_c, factor=4)
    O_l_m_c_o_o_i, O_l_m_c_o_i = s[O_l].split(O_l_m_c_o_i, factor=64)

    O_l_n_c_o_i, O_l_n_c_i = s[O_l].split(O_l_n_c, factor=64)
    O_l_n_c_o_o_i, O_l_n_c_o_i = s[O_l].split(O_l_n_c_o_i, factor=4)
    O_l_n_c_o_o_o, O_l_n_c_o_o_i = s[O_l].split(O_l_n_c_o_o_i, factor=8)

    O_l_k_o, O_l_k_i = s[O_l].split(O_l_k, factor=64)
    s[O_l].reorder(O_l_m_c_o_o_i, O_l_n_c_o_o_o, O_l_n_c_o_o_i, O_l_k_o, O_l_m_c_o_i, O_l_n_c_o_i, O_l_k_i, O_l_m_c_i, O_l_n_c_i)

    O_m, O_n = tuple(O.op.axis)
    O_m_o_i, O_m_i = s[O].split(O_m, factor=256)

    O_n_o_i, O_n_i = s[O].split(O_n, factor=256)
    O_n_o_o, O_n_o_i = s[O].split(O_n_o_i, factor=8)
    s[O].reorder(O_m_o_i, O_n_o_o, O_n_o_i, O_m_i, O_n_i)
    # s[O_l].compute_at(s[O], O_n_o_i)

    # O_m_o_o_fused_n_o_o_fused = s[O].fuse(O_m_o_o, O_n_o_o)
    # s[O].parallel(O_m_o_o_fused_n_o_o_fused)


    # s[O_l].pragma(O_l_m_c_o_o_i,a "auto_unroll_max_step", 0)
    # s[O_l].pragma(O_l_m_c_o_o_i, "unroll_explicit", True)
    # s[O_l].unroll(O_l_k_i)
    # s[O_l].unroll(O_l_m_c_i)
    # s[O_l].vectorize(O_l_n_c_i)

    # s[O].mark_no_bounds_check()
    # s[O_l].mark_no_bounds_check()


    if cache_write_tensor is None: return [O.op, O_l.op]
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
