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
args = parser.parse_args()

args.target = run_utils.get_arm_target()

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')

if args.no_raggedness:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [], [], lambda : lambda : MAX_LEN)
else:
    def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
lufw1 = len_ufw('s1_1', 1)
lufw32 = len_ufw('s2_32', 16)
lufw64 = len_ufw('s64', 64)

ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufw1.get_uf(),
    3: lufw32.get_uf(),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
if args.layout_unfused:
    width_ufs=[ls[0], lufw32.get_uf(), ls[1], lufw32.get_uf()]
else:
    width_ufs=[ls[0], lufw64.get_uf(), ls[1], lufw64.get_uf()]
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)


loop_ufs=[ls[0], ls[2], ls[1]]
Amax = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS), [bd, s1, md], loop_ufs,
                         lambda ds, rds: tvm.max(A[ds[bd], ds[s1], ds[md], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Amax', reduce_axis_ufs = [('k', lufw32.get_uf())])

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
Aexp = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                         lambda ds: tvm.exp((A[ds[bd], ds[s1], ds[md], ds[s2]] -
                                             Amax[ds[bd], ds[s1], ds[md]]) * scale), name = 'Aexp')

loop_ufs=[ls[0], ls[2], ls[1]]
Asum = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS), [bd, s1, md], loop_ufs,
                         lambda ds, rds: tvm.sum(Aexp[ds[bd], ds[s1], ds[md], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Asum', reduce_axis_ufs = [('k', lufw32.get_uf())])

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: Aexp[ds[bd], ds[s1], ds[md], ds[s2]] / Asum[ds[bd], ds[s1], ds[md]],
                      name = 'O', width_uf_lists=None if args.dense_storage else [width_ufs])

s = tvm.create_schedule([O.op])

if True:
    b, l1, h, l2 = s[O].leaf_iter_vars
    f = s[O].fuse(b, l1, padding=16)
    fo, fi = s[O].split(f, factor=16)
    s[O].parallel(fi)

    vo, vi = s[O].split(l2, factor=8)
    s[O].vectorize(vi)

    Al = s.cache_read(A, 'local', [Amax, Aexp])
    vo, vi = s[Al].split(s[Al].leaf_iter_vars[3], factor=8)
    s[Al].vectorize(vi)

    vo, vi = s[Aexp].split(s[Aexp].leaf_iter_vars[3], factor=8)
    s[Aexp].vectorize(vi)

    vo, vi = s[Amax].split(s[Amax].leaf_iter_vars[3], factor=8)
    s[Amax].unroll(vi)

    vo, vi = s[Asum].split(s[Asum].leaf_iter_vars[3], factor=8)
    s[Asum].unroll(vi)

    s[Al].compute_at(s[O], fi)
    s[Asum].compute_at(s[O], fi)
    s[Aexp].compute_at(s[O], fi)
    s[Amax].compute_at(s[O], fi)

    inputs = [[lens], [BS_VAR, A, O]]
else:
    inputs = [[lens], [BS_VAR, A, O, Amax, Aexp, Asum]]


def size_fn(l_inputs):
    if args.no_raggedness: return {}
    else:
        lens = l_inputs[0]
        if args.layout_unfused: fn = lufw32.get_fn(lens)
        else: fn = lufw64.get_fn(lens)
        return {
            A: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (fn(b) * fn(b))),
            O: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (fn(b) * fn(b)))
        }

prep_code_mode = 'no_prep_code' if args.no_raggedness else 'with_prep_code'
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=16,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR),
                                        prep_code_mode=prep_code_mode)

# out = out[2]
# ctr = 0
# out = out.flatten()
# for length in batches[0]:
#     rounded = utils.ceilmult(length, 32)
#     this_extent = utils.ceilmult(length, 32)
#     this_storage_extent = utils.ceilmult(length, 64) * utils.ceilmult(length, 64) * NUM_HEADS
#     print(length, rounded, 1 / rounded, np.mean(out[ctr:ctr + this_extent]))
#     ctr += this_storage_extent
