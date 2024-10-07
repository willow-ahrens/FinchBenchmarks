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
parser.add_argument('--sched', dest='sched', default=3, type=int)
args = parser.parse_args()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
IN_SIZE = 2048
OUT_SIZE = 512
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
id = Dim('id')
od = Dim('od')

def len_ufw(name): return Ufw(name, "l", (1, MAX_LEN), [bd], [lens], lambda lens: lambda b: lens[b])
lufw = len_ufw('s1')

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: lufw.get_uf(),
    2: Uf.from_constant('id', IN_SIZE, "l"),
    3: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((IN_SIZE, OUT_SIZE), name='W')
B = te.placeholder((OUT_SIZE,), name='B')

loop_ufs=[ls[0], ls[1], ls[3]]
width_ufs=loop_ufs
A2 = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                           name='A2', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3]]
width_ufs=None if args.dense_storage else [loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(W[k, ds[od]] * A[ds[bd], ds[s1], k], axis = k, dimensions = [id]),
                      name = 'S', width_uf_lists=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3]]
width_ufs=None if args.dense_storage else [[ls[0], lufw.get_uf(), ls[3]]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: S[ds[bd], ds[s1], ds[od]] + A2[ds[bd], ds[s1], ds[od]] + B[ds[od]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    if args.sched == 1:
        # O_local, = s.cache_write([O], "local", storage_layout_mode='loop_layout')
        O_local = S

        b, l, o, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
        l = s[O_local].fuse(b, l, padding = 2)
        loi, li = s[O_local].split(l, factor=2)

        ooi, oi = s[O_local].split(o, factor=2)

        koi, ki = s[O_local].split(k, factor=4)
        koo, koi = s[O_local].split(koi, factor=2)

        s[O_local].reorder(koo, koi, loi, ooi, ki, li, oi)

        if not args.debug_code:
            s[O_local].unroll(koi)
            s[O_local].unroll(loi)
            s[O_local].unroll(ooi)
            s[O_local].unroll(ki)
            s[O_local].unroll(li)
            s[O_local].unroll(oi)

        O_b, O_l, O_o = tuple(O.op.axis) + tuple(O.op.reduce_axis)
        O_l = s[O].fuse(O_b, O_l, padding = 32)

        O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
        O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=2)
        O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

        O_o_o_i, O_o_i = s[O].split(O_o, factor=4)
        O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
        O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
        s[O].reorder(O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)
        s[O].vectorize(O_o_i)

        A_shared = s.cache_read(A, "shared", [O_local])
        A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
        A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
        s[A_shared].compute_at(s[O_local], koo)

        W_shared = s.cache_read(W, "shared", [O_local], vanilla=True)
        W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
        s[W_shared].compute_at(s[O_local], koo)

        B_shared = s.cache_read(B, "shared", [O], vanilla=True)
        B_shared_ax0, = tuple(B_shared.op.axis)

        s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.y"))
        s[O].bind(O_o_o_o_o, te.thread_axis("blockIdx.x"))
        O_l_o_o_i_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
        s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
        O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
        s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
        s[O_local].compute_at(s[O], O_l_o_i_o_o_i_fused)
        s[B_shared].compute_at(s[O], O_l_o_i_o_o_i_fused)

        A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
        A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
        if not args.debug_functions: s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
        A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
        s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))
        s[A_shared].mark_no_bounds_check()

        W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
        W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=4)
        if not args.debug_functions: s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
        W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=32)
        s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        B_shared_ax0_o, B_shared_ax0_i = s[B_shared].split(B_shared_ax0, nparts=32)
        if not args.debug_functions: s[B_shared].vectorize(B_shared_ax0_i)
        s[B_shared].bind(B_shared_ax0_o, te.thread_axis("threadIdx.x"))

        s.fuse_tensor_dimensions(O_local, 0, 1)
        s.fuse_tensor_dimensions(A_shared, 0, 1)

        s[O_local].set_scope('local')

        s[O].mark_no_bounds_check()
        s[O_local].mark_no_bounds_check()
    elif args.sched == 2:
        S_b, S_l, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)
        S_l = s[S].fuse(S_l, S_b)

        S_k_o_i, S_k_i = s[S].split(S_k, factor=32)
        s[S].reorder(S_k_o_i, S_l, S_k_i, S_o)

        if not args.debug_code:
            s[S].unroll(S_o)

        O_b, O_l, O_o = tuple(O.op.axis) + tuple(O.op.reduce_axis)
        O_l = s[O].fuse(O_b, O_l, padding=32)
        O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
        O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=4)

        O_o_o_i, O_o_i = s[O].split(O_o, factor=2)
        O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=32)

        s[O].reorder(O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)
        s[O].vectorize(O_o_i)

        A_shared = s.cache_read(A, "shared", [S])
        A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
        A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
        s[A_shared].compute_at(s[S], S_k_o_i)

        W_shared = s.cache_read(W, "shared", [S], vanilla=True)
        W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
        s[W_shared].compute_at(s[S], S_k_o_i)

        O_l_o_o_i_o_o_o_o_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
        s[O].bind(O_l_o_o_i_o_o_o_o_fused, te.thread_axis("blockIdx.x"))
        O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
        s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
        s[S].compute_at(s[O], O_l_o_i_o_o_i_fused)

        A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
        A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=4)
        s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
        A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=128)
        s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
        W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=2)
        s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
        W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=128)
        s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        # s[S].pragma(S_l_o_o_o_o, "auto_unroll_max_step", 512)
        # s[S].pragma(S_l_o_o_o_o, "unroll_explicit", True)

        s.fuse_tensor_dimensions(S, 0, 1)
        s.fuse_tensor_dimensions(A_shared, 0, 1)

        s[S].mark_no_bounds_check()
        s[O].mark_no_bounds_check()

        s[S].set_scope('local')
    elif args.sched == 3:
        S_b, S_l, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)
        S_l = s[S].fuse(S_b, S_l)

        S_k_o_i, S_k_i = s[S].split(S_k, factor=4)
        S_k_o_o, S_k_o_i = s[S].split(S_k_o_i, factor=16)
        s[S].reorder(S_k_o_o, S_l, S_k_o_i, S_o, S_k_i)

        O_b, O_l, O_o = tuple(O.op.axis) + tuple(O.op.reduce_axis)
        O_l = s[O].fuse(O_b, O_l, padding=8)

        O_l_o_o_i, O_l_o_i = s[O].split(O_l, factor=2)
        O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=4)

        O_o_o_i, O_o_i = s[O].split(O_o, factor=2)
        O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)

        s[O].reorder(O_l_o_o_o, O_o_o_o_i, O_l_o_o_i, O_l_o_i, O_o_o_i, O_o_i)
        s[O].vectorize(O_o_i)

        A_shared = s.cache_read(A, "shared", [S])
        A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
        A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
        s[A_shared].compute_at(s[S], S_k_o_o)

        W_shared = s.cache_read(W, "shared", [S], vanilla=True)
        W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
        s[W_shared].compute_at(s[S], S_k_o_o)

        s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.y"))
        s[O].bind(O_o_o_o_i, te.thread_axis("blockIdx.x"))
        O_l_o_o_i_o_o_o_i_fused = O_l_o_o_i
        s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
        O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
        s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
        s[S].compute_at(s[O], O_l_o_i_o_o_i_fused)

        A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
        A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
        s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
        A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
        s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
        W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=2)
        s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
        W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=32)
        s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        s.fuse_tensor_dimensions(S, 0, 1)
        s.fuse_tensor_dimensions(A_shared, 0, 1)

        s[S].mark_no_bounds_check()
        s[O].mark_no_bounds_check()

        s[S].set_scope('local')
        # s[S].pragma(S_l_o_o_o_o, "auto_unroll_max_step", 512)
        # s[S].pragma(S_l_o_o_o_o, "unroll_explicit", True)
    elif args.sched == 4:
        S_b, S_l, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)
        S_l = s[S].fuse(S_b, S_l, padding = 2)

        S_k_o_i, S_k_i = s[S].split(S_k, factor=2)
        S_k_o_o, S_k_o_i = s[S].split(S_k_o_i, factor=16)
        s[S].reorder(S_k_o_o, S_k_o_i, S_o, S_k_i, S_l)

        if not args.debug_code:
            s[S].unroll(S_o)
            s[S].unroll(S_l)
            s[S].unroll(S_k_i)

        O_b, O_l, O_o = tuple(O.op.axis) + tuple(O.op.reduce_axis)
        O_l = s[O].fuse(O_b, O_l, padding = 32)

        O_l_o_i, O_l_i = s[O].split(O_l, factor=4)
        O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=4)
        O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)

        O_o_o_i, O_o_i = s[O].split(O_o, factor=2)
        O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
        s[O].reorder(O_l_o_o_o, O_o_o_o_i, O_l_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)
        s[O].vectorize(O_o_i)

        A_shared = s.cache_read(A, "shared", [S])
        A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
        A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
        s[A_shared].compute_at(s[S], S_k_o_o)

        W_shared = s.cache_read(W, "shared", [S], vanilla=True)
        W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
        s[W_shared].compute_at(s[S], S_k_o_o)

        # B_shared = s.cache_read(B, "shared", [O], vanilla=True)
        # B_shared_ax0, = tuple(B_shared.op.axis)

        # O_l_o_o_o_o_i_o_o_fused = s[O].fuse(O_l_o_o_o, O_o_o_o_i)
        s[O].bind(O_l_o_o_o, te.thread_axis("blockIdx.y"))
        s[O].bind(O_o_o_o_i, te.thread_axis("blockIdx.x"))
        O_l_o_o_i_o_o_o_i_fused = O_l_o_o_i
        s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
        O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
        s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
        s[S].compute_at(s[O], O_l_o_i_o_o_i_fused)

        # s[B_shared].compute_at(s[O], O_l_o_i_o_o_i_fused)
        # B_shared_ax0, = tuple(B_shared.op.axis)
        # B_shared_ax0_o, B_shared_ax0_i = s[B_shared].split(B_shared_ax0, factor=64)
        # s[B_shared].bind(B_shared_ax0_i, te.thread_axis("threadIdx.x"))

        A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
        A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=4)
        s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
        A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=64)
        s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
        W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=2)
        s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
        W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=64)
        s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        # s[S].pragma(S_l_o_o_o_o, "auto_unroll_max_step", 512)
        # s[S].pragma(S_l_o_o_o_o, "unroll_explicit", True)

        s[S].set_scope('local')
        s.fuse_tensor_dimensions(S, 0, 1)
        s.fuse_tensor_dimensions(A_shared, 0, 1)
        s[S].mark_no_bounds_check()
        s[O].mark_no_bounds_check()
        s[A_shared].mark_no_bounds_check()
    else:
        S_b, S_l, S_o, S_k = tuple(S.op.axis) + tuple(S.op.reduce_axis)
        S_l = s[S].fuse(S_b, S_l)

        S_k_o_i, S_k_i = s[S].split(S_k, factor=16)
        s[S].reorder(S_k_o_i, S_l, S_k_i, S_o)

        if not args.debug_code:
            s[S].unroll(S_k_i)
            s[S].unroll(S_o)

        O_b, O_l, O_o = tuple(O.op.axis) + tuple(O.op.reduce_axis)
        O_l = s[O].fuse(O_b, O_l, padding = 32)

        O_l_o_i, O_l_i = s[O].split(O_l, factor=4)
        O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=8)

        O_o_o_i, O_o_i = s[O].split(O_o, factor=2)
        O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
        O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=2)
        s[O].reorder(O_l_o_o_i, O_o_o_o_o, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)
        s[O].vectorize(O_o_i)

        A_shared = s.cache_read(A, "shared", [S])
        A_shared_axm1, A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
        A_shared_ax0 = s[A_shared].fuse(A_shared_axm1, A_shared_ax0)
        s[A_shared].compute_at(s[S], S_k_o_i)

        W_shared = s.cache_read(W, "shared", [S], vanilla=True)
        W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
        s[W_shared].compute_at(s[S], S_k_o_i)

        O_l_o_o_o_o_o_o_o_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_o)
        s[O].bind(O_l_o_o_o_o_o_o_o_fused, te.thread_axis("blockIdx.x"))
        O_l_o_o_i_o_o_o_i_fused = O_o_o_o_i
        s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
        O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
        s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
        s[S].compute_at(s[O], O_l_o_i_o_o_i_fused)

        A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
        A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
        s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
        A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=128)
        s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
        W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=2)
        s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
        W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=128)
        s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

        # s[S].pragma(S_l_o_o_o_o, "auto_unroll_max_step", 1024)
        # s[S].pragma(S_l_o_o_o_o, "unroll_explicit", True)

        s[S].set_scope('local')
        s.fuse_tensor_dimensions(S, 0, 1)
        s.fuse_tensor_dimensions(A_shared, 0, 1)
        s[S].mark_no_bounds_check()
        s[O].mark_no_bounds_check()
        s[A_shared].mark_no_bounds_check()


    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
    inputs = [[lens], [BS_VAR, A, A2, W, B, O]]
else:
    inputs = [[lens], [BS_VAR, A, A2, W, B, S, O]]

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: IN_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        O: OUT_SIZE * (BATCH_SIZE * MAX_LEN if args.dense_storage else
                       run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)))
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=32,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

# O  = out[-1]
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * OUT_SIZE
#     print(length, run_utils.stats(O[ctr:ctr + this_extent]))
#     ctr += this_extent
