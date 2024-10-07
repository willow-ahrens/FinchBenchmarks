import os
import numpy as np
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
parser.add_argument('--padding-mode', dest='padding_mode', default='remove')
args = parser.parse_args()

BATCH_SIZE = te.var('bs')
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1_1', 1)
lufw32 = len_ufw('s2_32', 32)
lufw64 = len_ufw('s64', 64)

if args.padding_mode == 'remove':
    a_fnw = lufw64
    o_fnw = lufw32
else:
    a_fnw = lufw32
    o_fnw = lufw64

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufw1.get_uf(),
    # 3: lufw32.get_uf(),
    3: Uf.from_constant('ml', MAX_LEN, 'l'),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
width_ufs=[ls[0], a_fnw.get_uf(), ls[1], a_fnw.get_uf()]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)

width_ufs=[ls[0], o_fnw.get_uf(), ls[1], o_fnw.get_uf()]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: A[ds[bd], ds[s1], ds[md], ds[s2]],
                      fpred = lambda ds: ds[s2] < utils.ceilmult(lens[ds[bd]], 32),
                      name = 'O', width_uf_lists=[width_ufs])

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    thread_x = tvm.thread_axis("threadIdx.x")
    thread_y = tvm.thread_axis("threadIdx.y")
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")

    b, s1, h, s2 = s[O].leaf_iter_vars
    f = s[O].fuse(b, s1)
    s[O].bind(f, block_x)

    # xo, xi = s[O].split(s2, factor = 32)
    # s[O].bind(xi, thread_x)
    s[O].bind(s2, thread_x)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))


def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (a_fnw.get_fn(lens)(b) * a_fnw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (o_fnw.get_fn(lens)(b) * o_fnw.get_fn(lens)(b)))
    }

inputs = [[lens], [BATCH_SIZE, A, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE))

# out = out[2]
# ctr = 0
# out = out.flatten()
# for length in batches[0]:
#     rounded = utils.ceilmult(length, 32)
#     this_extent = utils.ceilmult(length, 32)
#     this_storage_extent = utils.ceilmult(length, 64) * utils.ceilmult(length, 64) * NUM_HEADS
#     print(length, rounded, 1 / rounded, np.mean(out[ctr:ctr + this_extent]))
#     ctr += this_storage_extent
