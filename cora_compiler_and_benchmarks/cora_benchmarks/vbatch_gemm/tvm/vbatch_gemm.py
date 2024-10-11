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

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--batch-sizes', dest='batch_sizes', nargs='+', default=[32], type=int)
parser.add_argument('--tile-size', dest='tile_size', default=128, type=int)
parser.add_argument('--no-hoist-loads', dest='no_hoist_loads', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
parser.add_argument('--data-file', nargs='?', default='random')

parser.add_argument('--m1', dest='m1', default=2, type=int)
parser.add_argument('--m2', dest='m2', default=1, type=int)
parser.add_argument('--n1', dest='n1', default=32, type=int)
parser.add_argument('--n2', dest='n2', default=4, type=int)
parser.add_argument('--k1', dest='k1', default=8, type=int)
parser.add_argument('--k2', dest='k2', default=8, type=int)
parser.add_argument('--fs', dest='fs', default=3, type=int)
args = parser.parse_args()

BATCH_SIZE = te.var('bs')

ms = te.placeholder((BATCH_SIZE,), name = 'ms', dtype = 'int32')
ns = te.placeholder((BATCH_SIZE,), name = 'ns', dtype = 'int32')
ks = te.placeholder((BATCH_SIZE,), name = 'ks', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')
MIN_DIM, MAX_DIM = 4*args.tile_size, 12*args.tile_size

def f_mufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ms], lambda b: lambda b: args.tile_size * ms[b])
def f_nufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ns], lambda b: lambda b: args.tile_size * ns[b])
def f_kufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ks], lambda b: lambda b: args.tile_size * ks[b])

mufw = f_mufw('m')
nufw = f_nufw('m')
kufw = f_kufw('m')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: mufw.get_uf(),
    2: nufw.get_uf(),
    3: kufw.get_uf(),
}

loop_ufs=[ls[0], ls[1], ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, kd], loop_ufs, name='A', width_ufs=None, dtype='float32')
loop_ufs=[ls[0], ls[3], ls[2]]
B = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, kd, nd], loop_ufs, name='B', width_ufs=None, dtype='float32')

loop_ufs=[ls[0], ls[1], ls[2]]
Op = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs, name='Op', width_ufs=None, dtype='float32')

loop_ufs=[ls[0], ls[1], ls[2]]
S = te.ragged_compute((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[md], rds['k']] * B[ds[bd], rds['k'], ds[nd]],
                                              axis=rds['k'], dimensions=[kd]),
                      name = 'S', reduce_axis_ufs = [('k', kufw.get_uf())], width_uf_lists=None)

alpha = 0.01
beta = 0.03
loop_ufs=[ls[0], ls[1], ls[2]]
O = te.ragged_compute((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs,
                      lambda ds: alpha*S[ds[bd], ds[md], ds[nd]] + beta*Op[ds[bd], ds[md], ds[nd]],
                      name = 'O', width_uf_lists=None)

s = tvm.create_schedule([O.op])

prep_code_mode='with_prep_code'
if args.target == "cuda":
    O_local = S
    O_l_b_c, O_l_m_c, O_l_n_c, O_l_k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)

    O_l_k_o_i, O_l_k_i = s[O_local].split(O_l_k, factor=4)
    O_l_k_o_o, O_l_k_o_i = s[O_local].split(O_l_k_o_i, factor=4)
    s[O_local].reorder(O_l_b_c, O_l_k_o_o, O_l_k_o_i, O_l_m_c, O_l_n_c, O_l_k_i)

    O_b, O_m, O_n = tuple(O.op.axis) + tuple(O.op.reduce_axis)

    O_m_o_i, O_m_i = s[O].split(O_m, factor=4)
    O_m_o_o_i, O_m_o_i = s[O].split(O_m_o_i, factor=4)
    O_m_o_o_o, O_m_o_o_i = s[O].split(O_m_o_o_i, factor=4)

    O_n_o_i, O_n_i = s[O].split(O_n, factor=2)
    O_n_o_o_i, O_n_o_i = s[O].split(O_n_o_i, factor=32)
    O_n_o_o_o, O_n_o_o_i = s[O].split(O_n_o_o_i, factor=2)

    s[O].reorder(O_b, O_m_o_o_o, O_n_o_o_o, O_m_o_o_i, O_n_o_o_i, O_m_o_i, O_n_o_i, O_m_i, O_n_i)
    s[O].vectorize(O_n_i)


    B_s = s.cache_read(B, "shared", [O_local])
    B_s_ax0, B_s_ax1, B_s_ax2 = tuple(B_s.op.axis)
    s[B_s].compute_at(s[O_local], O_l_k_o_o)

    A_s = s.cache_read(A, "shared", [O_local])
    A_s_ax0, A_s_ax1, A_s_ax2 = tuple(A_s.op.axis)
    s[A_s].compute_at(s[O_local], O_l_k_o_o)

    O_m_o_o_o_f_n_o_o_o_f = s[O].fuse(O_m_o_o_o, O_n_o_o_o)
    O_b_m_o_o_o_f_n_o_o_o_f = s[O].fuse(O_b, O_m_o_o_o_f_n_o_o_o_f)
    s[O].bind(O_b_m_o_o_o_f_n_o_o_o_f, te.thread_axis("blockIdx.x"))
    O_m_o_o_i_f_n_o_o_i_f = s[O].fuse(O_m_o_o_i, O_n_o_o_i)
    s[O].bind(O_m_o_o_i_f_n_o_o_i_f, te.thread_axis("vthread"), no_unroll_vthread=(args.debug_code is not None))
    O_m_o_i_f_n_o_i_f = s[O].fuse(O_m_o_i, O_n_o_i)
    s[O].bind(O_m_o_i_f_n_o_i_f, te.thread_axis("threadIdx.x"))
    s[O_local].compute_at(s[O], O_m_o_i_f_n_o_i_f)

    B_s_ax1_f_ax2_f = s[B_s].fuse(B_s_ax1, B_s_ax2)
    B_s_ax1_f_ax2_f_o, B_s_ax1_f_ax2_f_i = s[B_s].split(B_s_ax1_f_ax2_f, factor=4)
    s[B_s].vectorize(B_s_ax1_f_ax2_f_i)
    B_s_ax1_f_ax2_f_o_o, B_s_ax1_f_ax2_f_o_i = s[B_s].split(B_s_ax1_f_ax2_f_o, factor=128)
    s[B_s].bind(B_s_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

    A_s_ax1_f_ax2_f = s[A_s].fuse(A_s_ax1, A_s_ax2)
    A_s_ax1_f_ax2_f_o, A_s_ax1_f_ax2_f_i = s[A_s].split(A_s_ax1_f_ax2_f, factor=4)
    s[A_s].vectorize(A_s_ax1_f_ax2_f_i)
    A_s_ax1_f_ax2_f_o_o, A_s_ax1_f_ax2_f_o_i = s[A_s].split(A_s_ax1_f_ax2_f_o, factor=128)
    s[A_s].bind(A_s_ax1_f_ax2_f_o_i, te.thread_axis("threadIdx.x"))

    if not args.debug_code:
        s[O_local].pragma(O_l_b_c, "auto_unroll_max_step", 16)
        s[O_local].pragma(O_l_b_c, "unroll_explicit", True)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    O_local = S
    Bl = s.cache_read(B, 'local', [O_local], layouts='dense')
    Al = s.cache_read(A, 'local', [O_local], layouts='dense')

    All = s.cache_read(Al, 'local', [O_local], layouts='dense')
    # Bll = s.cache_read(Bl, 'local', [O_local], layouts='dense')

    b, m, n, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)

    m1, m2 = args.m1, args.m2
    n1, n2 = args.n1, args.n2
    k1, k2 = args.k1, args.k2

    if not (m1 * m2 <= 128 and n1 * n2 <= 128 and k1 * k2 <= 128):
        exit(0)

    mo, mii = s[O_local].split(m, factor=m1)
    mo, mi = s[O_local].split(mo, factor=m2)

    no, nii = s[O_local].split(n, factor=n1)
    no, ni = s[O_local].split(no, factor=n2)

    ko, ki = s[O_local].split(k, factor=k1)
    ko, kii = s[O_local].split(ko, factor=k2)
    s[O_local].reorder(ko, mo, no, ki, mi, ni, kii, mii, nii)
    s[O_local].vectorize(nii)
    s[Bl].compute_at(s[O_local], ko)
    s[Al].compute_at(s[O_local], ko)

    # s[Bll].compute_at(s[O_local], ni)
    s[All].compute_at(s[O_local], kii)
    s[All].unroll(s[All].leaf_iter_vars[1])

    # if not args.debug_code:
    s[O_local].unroll(kii)
    s[O_local].unroll(mii)

    O_b, O_m, O_n = tuple(O.op.axis) + tuple(O.op.reduce_axis)
    O_m_o_o, O_m_i = s[O].split(O_m, factor=128)
    O_m_i_o, O_m_i_i = s[O].split(O_m_i, factor=128)
    O_n_o_o, O_n_i = s[O].split(O_n, factor=128)
    O_n_i_o, O_n_i_i = s[O].split(O_n_i, factor=128)

    s[O].reorder(O_b, O_m_o_o, O_n_o_o, O_m_i_o, O_n_i_o, O_m_i_i, O_n_i_i)

    if args.fs == 1:
        s[O].parallel(O_b)
        prep_code_mode='no_prep_code'
    elif args.fs == 2:
        fused = s[O].fuse(O_b, O_m_o_o)
        s[O].parallel(fused)
    else:
        fused = s[O].fuse(O_b, O_m_o_o)
        fused = s[O].fuse(fused, O_n_o_o)
        s[O].parallel(fused)
    s[O_local].compute_at(s[O], O_n_i_o)

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: BATCH_SIZE * MAX_DIM * MAX_DIM,
        B: BATCH_SIZE * MAX_DIM * MAX_DIM,
        O: BATCH_SIZE * MAX_DIM * MAX_DIM,
    }

bO = tvm.tir.decl_buffer((BATCH_SIZE, MAX_DIM, MAX_DIM), name="bO")
binds = {Op: bO, O: bO}
if args.only_prep_code: prep_code_mode = 'only_prep_code'
inputs = [[ms, ns, ks], [BATCH_SIZE, A, B, bO]]
# inputs = [[ms, ns, ks], [A, B, bO]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                               # run_function=run_utils.run_vbatch_gemm2,
                               run_function=run_utils.get_vbatch_gemm_run_fn(BATCH_SIZE),
                               prep_code_mode=prep_code_mode, binds=binds,
                               hoist_loads=not args.no_hoist_loads)

# A, W, O  = out
# for i in range(BATCH_SIZE):
    # length = batches[0][i]
    # print(batches[0][i], np.mean(O[i,0:length,:]))
