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
args = parser.parse_args()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
IN_SIZE = 512
OUT_SIZE = 64
QKV_NUM = 3

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qkv = Dim('qkv')
bd = Dim('bd')
s1 = Dim('s1')
md = Dim('md')
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('qkv', QKV_NUM, "l"),
    1: Uf.from_constant('bd', BATCH_SIZE, "l"),
    2: Uf.from_constant('md', NUM_HEADS, "l"),
    3: lufw1.get_uf(),
    4: Uf.from_constant('id', IN_SIZE, "l"),
    5: Uf.from_constant('od', OUT_SIZE, "l"),
}

a_fnw = lufw64
o_fnw = lufw1

loop_ufs=[ls[0], ls[1], ls[3], ls[2], ls[5]]
width_ufs=[ls[0], ls[1], lufw1.get_uf(), ls[2], ls[5]]
A = te.ragged_placeholder((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od],
                          loop_ufs, name = 'A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[2], ls[5]]
width_ufs=[ls[0], ls[1], lufw64.get_uf(), ls[2], ls[5]]
O = te.ragged_compute((QKV_NUM, BATCH_SIZE, MAX_LEN, NUM_HEADS, OUT_SIZE), [qkv, bd, s1, md, od], loop_ufs,
                      lambda ds: A[ds[qkv], ds[bd], ds[s1], ds[md], ds[od]], name = 'O', width_uf_lists=[width_ufs])

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    tile = 64
    rtile = 8
    nt = 8
    ks = 64

    thread_x = lambda: tvm.thread_axis("threadIdx.x")
    thread_y = lambda: tvm.thread_axis("threadIdx.y")
    block_x = lambda: tvm.thread_axis("blockIdx.x")
    block_y = lambda: tvm.thread_axis("blockIdx.y")

    q, b, l, n, h = s[O].leaf_iter_vars
    y = s[O].fuse(b, l)
    s[O].bind(y, block_y())
    s[O].bind(q, block_x())
    s[O].bind(n, thread_y())
    s[O].bind(h, thread_x())

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw1.get_fn(lens)(b)),
        O: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: lufw64.get_fn(lens)(b)),
    }

inputs = [[lens], [BS_VAR, A, O]]
binds = {}

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds, pad_sum=tile,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

# _, A, W, B, O = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     this_extent = length * OUT_SIZE
#     print(length, np.mean(O[ctr:ctr + this_extent]))
#     ctr += this_extent
