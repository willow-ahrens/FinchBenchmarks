import numpy as np
import math
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
import utils
import run_utils

parser = run_utils.get_cmd_parser()
args = parser.parse_args()
args.full_dense = True

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

ufw = Ufw('s', 'l', (MAX_LEN, MAX_LEN), [], [], lambda: lambda: MAX_LEN)
lufw1 = ufw
lufw64 = ufw

ls =  {
    0: Uf.from_constant('qkv', QKV_NUM, 'l'),
    1: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    2: Uf.from_constant('md', NUM_HEADS, 'l'),
    3: lufw1.get_uf(),
    4: Uf.from_constant('id', IN_SIZE, 'l'),
    5: Uf.from_constant('od', OUT_SIZE, 'l'),
}

loop_ufs=[ls[1], ls[3], ls[4]]
width_ufs=None
QKV = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, IN_SIZE), [bd, s1, id], loop_ufs,
                            name='QKV', width_ufs=width_ufs)

W = te.placeholder((QKV_NUM, NUM_HEADS, OUT_SIZE, IN_SIZE), name='W')
B = te.placeholder((QKV_NUM, NUM_HEADS, OUT_SIZE), name='B')

loop_ufs=[ls[0], ls[1], ls[3], ls[2], ls[5]]
width_ufs=None
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
S = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od], loop_ufs,
                      lambda ds: tvm.sum(W[ds[qkv], ds[md], ds[od], k] * QKV[ds[bd], ds[s1], k],
                                         axis = k, dimensions = [id]),
                      name = 'S', width_uf_lists=width_ufs)

width_ufs=None
O = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od], loop_ufs,
                      lambda ds: S[ds[qkv], ds[bd], ds[s1], ds[md], ds[od]] + B[ds[qkv], ds[md], ds[od]],
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    # s.fuse_tensor_dimensions(QKV, 0, 1)

    O_local = S
    q_c, b_c, l_c, n_c, h_c, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
    l_c = s[O_local].fuse(b_c, l_c)
    l_coi, l_ci = s[O_local].split(l_c, factor=2)
    koo, koi = s[O_local].split(k, factor=8)
    s[O_local].reorder(q_c, h_c, koo, koi, l_coi, l_ci, n_c)
    s[O_local].mark_no_bounds_check()

    O_q, O_b, O_l, O_n, O_h = tuple(O.op.axis) + tuple(O.op.reduce_axis)

    O_loi, O_li = s[O].split(O_l, factor=8)
    O_looi, O_loi = s[O].split(O_loi, factor=4)
    O_looo, O_looi = s[O].split(O_looi, factor=2)

    O_noi, O_ni = s[O].split(O_n, factor=2)
    O_nooi, O_noi = s[O].split(O_noi, factor=2)

    O_hooi, O_hoi = s[O].split(O_h, factor=8)
    O_hooo, O_hooi = s[O].split(O_hooi, factor=2)

    s[O].reorder(O_q, O_looo, O_nooi, O_hooo, O_looi, O_hooi, O_loi, O_noi, O_hoi, O_li, O_ni)
    s[O].vectorize(O_ni)

    QKV_sh = s.cache_read(QKV, 'shared', [O_local])
    QKV_sh_ax00, QKV_sh_ax01, QKV_sh_ax1 = tuple(QKV_sh.op.axis)
    QKV_sh_ax0 = s[QKV_sh].fuse(QKV_sh_ax00, QKV_sh_ax01)
    s[QKV_sh].compute_at(s[O_local], koo)
    s[QKV_sh].mark_no_bounds_check()

    W_sh = s.cache_read(W, 'shared', [O_local], vanilla = True)
    W_sh_ax0, W_sh_ax1, W_sh_ax2, W_sh_ax3 = tuple(W_sh.op.axis)
    s[W_sh].compute_at(s[O_local], koo)

    s[O].bind(O_q, te.thread_axis('blockIdx.z'))
    O_looo = s[O].fuse(O_b, O_looo)
    s[O].bind(O_looo, te.thread_axis('blockIdx.y'))
    O_q_looo_f_nooo_f_hooo_f = s[O].fuse(O_nooi, O_hooo)
    s[O].bind(O_q_looo_f_nooo_f_hooo_f, te.thread_axis('blockIdx.x'))
    O_qooi_looi_f_nooi_f_hooi_f = s[O].fuse(O_looi, O_hooi)
    s[O].bind(O_qooi_looi_f_nooi_f_hooi_f, te.thread_axis('vthread'), no_unroll_vthread=args.debug_code)
    O_qoi_loi_f_noi_f_hoi_f = s[O].fuse(O_loi, O_noi, O_hoi)
    s[O].bind(O_qoi_loi_f_noi_f_hoi_f, te.thread_axis('threadIdx.x'))
    s[O_local].compute_at(s[O], O_qoi_loi_f_noi_f_hoi_f)

    B_sh = s.cache_read(B, 'shared', [O], vanilla = True)
    B_sh_ax0, B_sh_ax1, B_sh_ax2 = tuple(B_sh.op.axis)
    s[B_sh].compute_at(s[O], O_qoi_loi_f_noi_f_hoi_f)

    QKV_sh_ax0_ax1_f = s[QKV_sh].fuse(QKV_sh_ax0, QKV_sh_ax1)
    QKV_sh_ax0_ax1_f_o, QKV_sh_ax0_ax1_f_i = s[QKV_sh].split(QKV_sh_ax0_ax1_f, factor=4)
    s[QKV_sh].vectorize(QKV_sh_ax0_ax1_f_i)
    QKV_sh_ax0_ax1_f_o_o, QKV_sh_ax0_ax1_f_o_i = s[QKV_sh].split(QKV_sh_ax0_ax1_f_o, factor=64)
    s[QKV_sh].bind(QKV_sh_ax0_ax1_f_o_i, te.thread_axis('threadIdx.x'))

    W_sh_ax0_ax1_f_ax2_f_ax3_f = s[W_sh].fuse(W_sh_ax0, W_sh_ax1, W_sh_ax2, W_sh_ax3)
    W_sh_ax0_ax1_f_ax2_f_ax3_f_o, W_sh_ax0_ax1_f_ax2_f_ax3_f_i = s[W_sh].split(W_sh_ax0_ax1_f_ax2_f_ax3_f, factor=4)
    s[W_sh].vectorize(W_sh_ax0_ax1_f_ax2_f_ax3_f_i)
    W_sh_ax0_ax1_f_ax2_f_ax3_f_o_o, W_sh_ax0_ax1_f_ax2_f_ax3_f_o_i = s[W_sh].split(W_sh_ax0_ax1_f_ax2_f_ax3_f_o, factor=64)
    s[W_sh].bind(W_sh_ax0_ax1_f_ax2_f_ax3_f_o_i, te.thread_axis('threadIdx.x'))

    B_sh_ax0_ax1_f_ax2_f_ax3_f = s[B_sh].fuse(B_sh_ax0, B_sh_ax1, B_sh_ax2)
    B_sh_ax0_ax1_f_ax2_f_ax3_f_o_o, B_sh_ax0_ax1_f_ax2_f_ax3_f_o_i = s[B_sh].split(B_sh_ax0_ax1_f_ax2_f_ax3_f, factor=64)
    s[B_sh].bind(B_sh_ax0_ax1_f_ax2_f_ax3_f_o_i, te.thread_axis('threadIdx.x'))

    s[O_local].set_scope('local')
    s[O].mark_no_bounds_check()

    if not args.debug_code:
        s[O_local].pragma(q_c, 'auto_unroll_max_step', 512)
        s[O_local].pragma(q_c, 'unroll_explicit', True)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    s.fuse_tensor_dimensions(QKV, 0, 1)
    pass

def size_fn(l_inputs):
    return {}

if args.target == 'cuda':
    inputs = [[lens], [BS_VAR, QKV, W, B, O]]
else:
    inputs = [[lens], [BS_VAR, QKV, W, B, S, O]]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR),
                                        prep_code_mode='no_prep_code')

# q_size = 0
# for length in batches[0]:
#     q_size += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE

# ctr = 0
# O  = out[-1]
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * NUM_HEADS * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]), np.mean(O[ctr+q_size:ctr+q_size + this_extent]),
#           np.mean(O[ctr+2*q_size:ctr+2*q_size + this_extent]))
#     ctr += utils.ceilmult(length, 64) * NUM_HEADS * OUT_SIZE




# floordiv(
#     floormod(
#         blockIdx.y*64 + floordiv(vthread.s, 2)*32 + floordiv(threadIdx.x, 16)*8,
#         512) + 7,
#     512) +
# 1
