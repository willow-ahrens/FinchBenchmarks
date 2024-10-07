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
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), TILE)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

def lb(name): return Uf(name, "l", (0, MAX_LEN), [bd], lambda b: utils.floormult(lens[b], TILE))
def ub(name): return Uf(name, "l", (TILE, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], TILE))

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: Uf.from_constant('md', NUM_HEADS, "l"),
    2: lufw64.get_uf(),
    3: lufw64.get_uf(),
    4: Uf.from_constant('hd', HEAD_SIZE, "l"),
    5: Uf.from_constant('qk', 3, "l"),
}

loop_ufs=[ls[0], ls[2], ls[1], (lb('lb'), ub('ub'))]
width_ufs = None if args.dense_storage else [[ls[0], ls[2], ls[1], ls[3]]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: -float('inf'), name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == "cuda":
    thread_x = lambda: tvm.thread_axis("threadIdx.x")
    thread_y = lambda: tvm.thread_axis("threadIdx.y")
    block_x = lambda: tvm.thread_axis("blockIdx.x")
    block_y = lambda: tvm.thread_axis("blockIdx.y")

    b, l, m, h = s[O].op.axis
    l = s[O].fuse(b, l)
    s[O].bind(l, block_x())
    s[O].bind(m, thread_y())
    ho, hi = s[O].split(h, nparts=64)
    s[O].bind(ho, thread_x())

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
else:
    pass

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (lufw.get_fn(lens)(b) *
                                                       lufw.get_fn(lens)(b)))
    }

inputs = [[lens], [BATCH_SIZE, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE))

# _, O = out
# O = O.flatten()
# ctr = 0
# for length in batches[0]:
#     roundedu = int(utils.ceilmult(length, TILE))
#     roundedd = int(utils.floormult(length, TILE))
#     this_storage_extent = roundedu * NUM_HEADS * OUT_SIZE
#     # print(type(length), type(roundedd), type(roundedu))
#     if roundedd != length and roundedd > 0:
#         print(length, roundedd, roundedu,
#               np.mean(O[ctr:ctr+roundedd*NUM_HEADS*OUT_SIZE]),
#               np.mean(O[ctr+length*NUM_HEADS*OUT_SIZE:ctr+roundedu*NUM_HEADS*OUT_SIZE]))
#     ctr += this_storage_extent
