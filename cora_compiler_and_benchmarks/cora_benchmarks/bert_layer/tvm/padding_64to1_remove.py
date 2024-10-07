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
OUT_SIZE = 512

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
s1 = Dim('s1')
od = Dim('od')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1', 1)
lufw64 = len_ufw('s2', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: lufw1.get_uf(),
    2: Uf.from_constant('od', OUT_SIZE, 'l'),
}

a_fnw = lufw64
o_fnw = lufw1

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=[ls[0], a_fnw.get_uf(), ls[2]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs, name = 'A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2]]
width_ufs=[ls[0], o_fnw.get_uf(), ls[2]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: A[ds[bd], ds[s1], ds[od]], name = 'O', width_uf_lists=[width_ufs])

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

    b, l, h = s[O].leaf_iter_vars
    y = s[O].fuse(b, l)
    s[O].bind(y, block_x())
    s[O].bind(h, thread_x())

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: a_fnw.get_fn(lens)(b)),
        O: OUT_SIZE * run_utils.prefix_sum(len(lens), lambda b: o_fnw.get_fn(lens)(b)),
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
