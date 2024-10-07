import math
import os
import utils
import run_utils
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--peel-loops', dest='peel_loops', default=False, action='store_true')
parser.add_argument('--unroll-loops', dest='unroll_loops', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

IN_SIZE = 2048
OUT_SIZE = 512
MAX_LEN = 12288 # utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 64)

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

s = Dim('s')
id = Dim('id')
od = Dim('od')

ls =  {
    0: Uf.from_constant('s', MAX_LEN, "l"),
    1: Uf.from_constant('id', IN_SIZE, "l"),
    2: Uf.from_constant('od', OUT_SIZE, "l"),
}

loop_ufs=[ls[0], ls[1]]
width_ufs=loop_ufs
A = te.ragged_placeholder((MAX_LEN, IN_SIZE), [s, id], loop_ufs, name='A', width_ufs=width_ufs)
W = te.placeholder((IN_SIZE, OUT_SIZE), name='W')

loop_ufs=[ls[0], ls[2]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, IN_SIZE), name = 'k')
O = te.ragged_compute((MAX_LEN, OUT_SIZE), [s, od], loop_ufs,
                      lambda ds: tvm.sum(W[k, ds[od]] * A[ds[s], k], axis = k, dimensions = [id]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

O_local, = s.cache_write([O], "local")
l, o, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)
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

O_l, O_o, O_k = tuple(O.op.axis) + tuple(O.op.reduce_axis)
O_l_o_i, O_l_i = s[O].split(O_l, factor=8)
O_l_o_o_i, O_l_o_i = s[O].split(O_l_o_i, factor=2)
O_l_o_o_o, O_l_o_o_i = s[O].split(O_l_o_o_i, factor=2)
O_o_o_i, O_o_i = s[O].split(O_o, factor=4)
O_o_o_o_i, O_o_o_i = s[O].split(O_o_o_i, factor=16)
O_o_o_o_o, O_o_o_o_i = s[O].split(O_o_o_o_i, factor=1)
s[O].reorder(O_l_o_o_o, O_o_o_o_o, O_l_o_o_i, O_o_o_o_i, O_l_o_i, O_o_o_i, O_l_i, O_o_i)

A_shared = s.cache_read(A, "shared", [O_local], vanilla=True)
A_shared_ax0, A_shared_ax1 = tuple(A_shared.op.axis)
s[A_shared].compute_at(s[O_local], koo)

W_shared = s.cache_read(W, "shared", [O_local], vanilla=True)
W_shared_ax0, W_shared_ax1 = tuple(W_shared.op.axis)
s[W_shared].compute_at(s[O_local], koo)

O_l_o_o_o_o_o_o_o_fused = s[O].fuse(O_l_o_o_o, O_o_o_o_o)
s[O].bind(O_l_o_o_o_o_o_o_o_fused, te.thread_axis("blockIdx.x"))
O_l_o_o_i_o_o_o_i_fused = s[O].fuse(O_l_o_o_i, O_o_o_o_i)
s[O].bind(O_l_o_o_i_o_o_o_i_fused, te.thread_axis("vthread"))
O_l_o_i_o_o_i_fused = s[O].fuse(O_l_o_i, O_o_o_i)
s[O].bind(O_l_o_i_o_o_i_fused, te.thread_axis("threadIdx.x"))
s[O_local].compute_at(s[O], O_l_o_i_o_o_i_fused)

A_shared_ax0_ax1_fused = s[A_shared].fuse(A_shared_ax0, A_shared_ax1)
A_shared_ax0_ax1_fused_o, A_shared_ax0_ax1_fused_i = s[A_shared].split(A_shared_ax0_ax1_fused, factor=2)
# s[A_shared].vectorize(A_shared_ax0_ax1_fused_i)
A_shared_ax0_ax1_fused_o_o, A_shared_ax0_ax1_fused_o_i = s[A_shared].split(A_shared_ax0_ax1_fused_o, factor=32)
s[A_shared].bind(A_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

W_shared_ax0_ax1_fused = s[W_shared].fuse(W_shared_ax0, W_shared_ax1)
W_shared_ax0_ax1_fused_o, W_shared_ax0_ax1_fused_i = s[W_shared].split(W_shared_ax0_ax1_fused, factor=4)
# s[W_shared].vectorize(W_shared_ax0_ax1_fused_i)
W_shared_ax0_ax1_fused_o_o, W_shared_ax0_ax1_fused_o_i = s[W_shared].split(W_shared_ax0_ax1_fused_o, factor=32)
s[W_shared].bind(W_shared_ax0_ax1_fused_o_i, te.thread_axis("threadIdx.x"))

suffix = ""
gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0] + suffix
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

# inputs = [[], [M, A, W, O]]
# if args.debug_code:
#     lowered = tvm.lower(s, inputs, args.target, simple_mode = True)
#     print(lowered)
#     # fadd, _ = tvm.build(s, inputs, args.target)
#     # if args.target == 'cuda':
#         # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
#     # else:
#         # print('-----CPU code-----\n' + fadd.get_source())
# else:
#     fadd, _ = tvm.build(s, inputs, args.target)
#     ctx = tvm.gpu(0)
#     evaluator = fadd.time_evaluator(fadd.entry_name, ctx, number=10, repeat=10)
#     inputs = [tvm.nd.array(run_utils.create_numpy_array(t, "float32", {M:TL}), ctx) for t in inputs[1][1:]]
#     inputs = [TL] + inputs
#     eval_result = evaluator(*inputs)
#     print(eval_result.mean * 1000)


inputs = [[], [A, W, O]]
if args.debug_code:
    lowered = tvm.lower(s, inputs, args.target, simple_mode = True)
    print(lowered)
    # fadd, _ = tvm.build(s, inputs, args.target)
    # if args.target == 'cuda':
        # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
    # else:
        # print('-----CPU code-----\n' + fadd.get_source())
else:
    fadd, _ = tvm.build(s, inputs, args.target)
    ctx = tvm.gpu(0)
    evaluator = fadd.time_evaluator(fadd.entry_name, ctx, number=10, repeat=10)
    inputs = [tvm.nd.array(run_utils.create_numpy_array(t, "float32"), ctx) for t in inputs[1]]
    inputs = inputs
    # fadd(*inputs)
    eval_result = evaluator(*inputs)
    print(eval_result.mean * 1000)
