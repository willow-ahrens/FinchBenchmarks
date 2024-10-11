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
args = parser.parse_args()

BATCH_SIZE = te.var('bs')
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
OUT_SIZE = 512

eps = 0.001
beta = 0.2
gamma = 0.5

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))

lufw = len_ufw('s', 1)
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: lufw.get_uf(),
    2: Uf.from_constant('od', OUT_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                          name='A', width_ufs=width_ufs)

B = te.placeholder((OUT_SIZE, ), name='B')
G = te.placeholder((OUT_SIZE, ), name='G')

loop_ufs=[ls[0], ls[1]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, OUT_SIZE), name = 'k')
Am1 = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, s1], loop_ufs,
                        lambda ds: tvm.sum(A[ds[bd], ds[s1], k], axis=k, dimensions=[od]),
                        name = 'Am1')

loop_ufs=[ls[0], ls[1]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, OUT_SIZE), name = 'k')
Am2 = te.ragged_compute((BATCH_SIZE, MAX_LEN), [bd, s1], loop_ufs,
                        lambda ds: tvm.sum((A[ds[bd], ds[s1], k] - Am1[ds[bd], ds[s1]]) *
                                           (A[ds[bd], ds[s1], k] - Am1[ds[bd], ds[s1]]),
                                           axis=k, dimensions=[od]),
                        name = 'Am2')

def compute_body(ds):
    mean1 = Am1[ds[bd], ds[s1]]/OUT_SIZE
    mean2 = Am2[ds[bd], ds[s1]]/OUT_SIZE
    std = tvm.sqrt(mean2 - mean1*mean1 + eps)
    normed = (A[ds[bd], ds[s1], ds[od]] - mean1) / (std + 1e-5)
    return B[ds[od]] + G[ds[od]] * normed

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=None if args.dense_storage else [loop_ufs]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs, compute_body, name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")

    ntx = 32
    ko, ki = s[Am1].split(s[Am1].op.reduce_axis[0], factor = ntx)
    Am1_rf = s.rfactor(Am1, ki, 1)

    ko, ki = s[Am2].split(s[Am2].op.reduce_axis[0], factor = ntx)
    Am2_rf = s.rfactor(Am2, ki, 1)

    b, l, h = s[O].leaf_iter_vars
    f = s[O].fuse(b, l)
    s[O].bind(f, block_x)


    ho, hi = s[O].split(h, factor = ntx)
    s[O].bind(hi, thread_x)

    s[Am1_rf].compute_at(s[Am1], s[Am1].leaf_iter_vars[2])
    s[Am2_rf].compute_at(s[Am2], s[Am2].leaf_iter_vars[2])
    s[Am1].compute_at(s[O], f)
    s[Am2].compute_at(s[O], f)

    s[Am1].bind(s[Am1].op.reduce_axis[0], thread_x)
    s[Am2].bind(s[Am2].op.reduce_axis[0], thread_x)

    s[Am1_rf].set_scope('local')
    s[Am2_rf].set_scope('local')
    s[Am1].set_scope('local')
    s[Am2].set_scope('local')

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
    inputs = [[lens], [BATCH_SIZE, A, B, G, O]]
else:
    inputs = [[lens], [BATCH_SIZE, A, B, G, O, Am1, Am2]]


def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        # A: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
        # O: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw.get_fn(lens)(b)),
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE))

_, A, B, G, O = out[0:5]
ctr = 0
O = O.flatten()
# for i in range(A1.shape[0]):
#     this_a1 = A1[i]
#     this_a2 = A2[i]
#     added = this_a1 + this_a2
#     mean = np.mean(added, axis=1, keepdims=True)
#     std = np.std(added, axis=1, keepdims=True)
#     res = beta + gamma * ((added - mean) / (std + eps))
#     length = batches[0][i]
#     this_extent = batches[0][i] * OUT_SIZE
#     print(length, np.mean(res), np.std(res), np.mean(O[ctr:ctr+this_extent]), np.std(O[ctr:ctr+this_extent]))
#     ctr += this_extent

# for length in batches[0]:
#     this_extent = length * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr+this_extent]), np.std(O[ctr:ctr+this_extent]))
#     ctr += this_extent
