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

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
NUM_HEADS = 8
HEAD_SIZE = 64
TILE1=64
if args.dataset in ['race', 'wiki_512', 'squadv2']: TILE2=64
else: TILE2=32
RTILE=4
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len1_ufw(name): return Ufw(name, "l", (TILE1, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], TILE1))
def len2_ufw(name): return Ufw(name, "l", (TILE2, MAX_LEN), [s1], [], lambda: lambda s: utils.ceilmult(s + 1, TILE2))
l1ufw = len1_ufw('s1')
l2ufw = len2_ufw('s2')


luf1 = l1ufw.get_uf()
luf2 = l2ufw.get_uf()
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: luf1,
    3: luf1,
    4: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    5: Uf.from_constant('qk', 3, "l"),
}

loop_ufs=[ls[5], ls[0], ls[2], ls[1], ls[4]]
width_ufs=[ls[5], ls[0], luf1, ls[1], ls[4]]
Q = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[5], ls[0], ls[3], ls[1], ls[4]]
width_ufs=[ls[5], ls[0], luf1, ls[1], ls[4]]
K = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[ls[0], luf1, ls[1], luf2]
width_ufs=[[ls[0], luf1, ls[1], luf1]]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[0, ds[bd], ds[s1], ds[md], k] * K[0, ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions = [hd]),
                      name = 'S', width_uf_lists=width_ufs)

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.if_then_else(ds[s2] >= ds[s1] + 1, -float('inf'), S[ds[bd], ds[s1], ds[md], ds[s2]]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if True:
    thread_x = lambda: tvm.thread_axis("threadIdx.x")
    thread_y = lambda: tvm.thread_axis("threadIdx.y")
    block_x = lambda: tvm.thread_axis("blockIdx.x")
    block_y = lambda: tvm.thread_axis("blockIdx.y")
    ntx = 8
    nty = 8

    s[S].set_scope('local')
    Ol = S
    Qs = s.cache_read(Q, "shared", [Ol], layouts='dense')
    Ks = s.cache_read(K, "shared", [Ol], layouts='dense')

    Ql = s.cache_read(Qs, "local", [Ol], layouts='dense')
    Kl = s.cache_read(Ks, "local", [Ol], layouts='dense')

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    yo, yi = s[O].split(y, factor = TILE2)
    xo, xi = s[O].split(x, factor = TILE2)

    ###############################################################################
    # b, x, h, y = s[O].leaf_iter_vars[0:4]

    # s[O].bind(b, block_x())
    # s[O].bind(h, block_y())
    # # xo, xi = s[O].split(x, factor = 64)
    # # s[O].bind(xi, thread_x())
    # yo, yi = s[O].split(y, factor = 64)
    # s[O].bind(yi, thread_y())
    ###############################################################################

    s[O].reorder(b, xo, yo, h, xi, yi)

    f1 = s[O].fuse(xo, yo)
    f = s[O].fuse(b, f1)
    s[O].bind(f, block_x())
    s[O].bind(h, block_y())

    xio, xii = s[O].split(xi, factor = ntx)
    yio, yii = s[O].split(yi, factor = nty)
    s[O].reorder(xii, yii, xio, yio)
    s[O].bind(xii, thread_y())
    s[O].bind(yii, thread_x())
    s[O].bind(xio, tvm.thread_axis("vthread", name='vth1'), no_unroll_vthread=args.debug_code)
    s[O].bind(yio, tvm.thread_axis("vthread", name='vth2'), no_unroll_vthread=args.debug_code)
    s[Ol].compute_at(s[O], yio)

    b, x, h, y, k = s[Ol].leaf_iter_vars
    s[Ol].reorder(h, k, x, y)
    ko, ki = s[Ol].split(k, nparts = 4)
    s[Qs].compute_at(s[Ol], ko)
    s[Ks].compute_at(s[Ol], ko)
    s[Ql].compute_at(s[Ol], ki)
    s[Kl].compute_at(s[Ol], ki)

    x, h, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3], s[Ks].leaf_iter_vars[4]
    s[Ks].reorder(h, y, x)
    f = s[Ks].fuse(x, y)
    fo, fi = s[Ks].split(f, factor = ntx * nty * 4)
    fio, fii = s[Ks].split(fi, factor = ntx * 4)
    fiio, fiii = s[Ks].split(fii, factor = 4)
    s[Ks].bind(fio, thread_y())
    s[Ks].bind(fiio, thread_x())
    s[Ks].vectorize(fiii)

    x, h, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3], s[Qs].leaf_iter_vars[4]
    s[Qs].reorder(h, y, x)
    f = s[Qs].fuse(x, y)
    fo, fi = s[Qs].split(f, factor = ntx * nty * 4)
    fio, fii = s[Qs].split(fi, factor = ntx * 4)
    fiio, fiii = s[Qs].split(fii, factor = 4)
    s[Qs].bind(fio, thread_y())
    s[Qs].bind(fiio, thread_x())
    s[Qs].vectorize(fiii)

    s.reorder_tensor_dimensions(Ks, 2, 3)
    s.reorder_tensor_dimensions(Ks, 3, 4)
    s.reorder_tensor_dimensions(Qs, 2, 3)
    s.reorder_tensor_dimensions(Qs, 3, 4)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))
    inputs = [[lens], [BS_VAR, Q, K, O]]
else:
    inputs = [[lens], [BS_VAR, Q, K, O, S]]

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        Q: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (l1ufw.get_fn(lens)(b))),
        K: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (l1ufw.get_fn(lens)(b))),
        O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            lambda b: (l1ufw.get_fn(lens)(b) *
                                                       l1ufw.get_fn(lens)(b)))
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64, hoist_loads=True,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

# O = out[3]
# O = O.flatten()
# ctr = 0
# for length in batches[0]:
    # rounded = utils.ceilmult(length, 64)
    # this_extent = rounded
    # this_storage_extent = rounded * rounded * NUM_HEADS
    # print(rounded, O[ctr], np.mean(O[ctr+length*rounded*NUM_HEADS:ctr+length*rounded*NUM_HEADS+length]))
    # ctr += this_storage_extent
