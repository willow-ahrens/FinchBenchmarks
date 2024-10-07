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
TILE1 = 64
TILE2 = 64
TILE3 = 16
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), max(TILE1, TILE2, TILE3))

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
def len3_ufw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [s1], [], lambda: lambda s: utils.ceilmult(s + 1, pad))
l1ufw = len_ufw('s1', TILE1)
l2ufw = len_ufw('s2', TILE2)
l3ufw = len3_ufw('s2k', TILE3)

luf1 = l1ufw.get_uf()
luf2 = l2ufw.get_uf()
luf3 = l3ufw.get_uf()
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: luf1,
    3: luf2,
    4: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    5: Uf.from_constant('qk', 3, 'l'),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[3]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[5], ls[0], ls[3], ls[1], ls[4]]
width_ufs=loop_ufs
V = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s2, md, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[2], ls[1], ls[4]]
width_ufs=[loop_ufs]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s1, md, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[s1], ds[md], rds['k']] *
                                              V(2, ds[bd], rds['k'], ds[md], ds[hd]),
                                              axis=rds['k'], dimensions = [s2]),
                      name = 'O', reduce_axis_ufs = [('k', luf3)],
                      width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    thread_x = lambda: tvm.thread_axis("threadIdx.x")
    thread_y = lambda: tvm.thread_axis("threadIdx.y")
    block_x = lambda: tvm.thread_axis("blockIdx.x")
    block_y = lambda: tvm.thread_axis("blockIdx.y")
    nty = 8
    ntx = 32
    # nty = 4
    # ntx = 16

    Ol = s.cache_write(O, "local")
    As = s.cache_read(A, "shared", [Ol])
    Vs = s.cache_read(V, "shared", [Ol])

    Al = s.cache_read(As, "local", [Ol])
    Vl = s.cache_read(Vs, "local", [Ol])

    b, x, h, y = s[O].leaf_iter_vars
    xo, xi = s[O].split(x, factor = 64)

    s[O].reorder(b, xo, h, y, xi)
    f = s[O].fuse(b, xo)
    s[O].bind(f, block_x())
    s[O].bind(h, block_y())
    s[Ol].compute_at(s[O], h)

    xio, xii = s[O].split(xi, nparts = nty)
    yo, yi = s[O].split(y, factor = ntx)
    s[O].reorder(xio, yi, yo, xii)
    s[O].bind(xio, thread_y())
    s[O].bind(yi, thread_x())
    s[O].bind(yo, tvm.thread_axis("vthread"))
    s[Ol].compute_at(s[O], yo)

    b, x, h, y, k = s[Ol].leaf_iter_vars
    s[Ol].reorder(b, h, k, x, y)
    ko, ki = s[Ol].split(k, factor = 16)
    s[As].compute_at(s[Ol], ko)
    s[Vs].compute_at(s[Ol], ko)
    s[Al].compute_at(s[Ol], ki)
    s[Vl].compute_at(s[Ol], ki)
    # s[Ol].peel(ko)

    b, x, h, y = s[As].leaf_iter_vars
    s[As].reorder(b, h, x, y)
    f = s[As].fuse(x, y)
    fo, fi = s[As].split(f, factor = ntx * nty * 4)
    fio, fii = s[As].split(fi, factor = ntx * 4)
    fiio, fiii = s[As].split(fii, factor = 4)
    s[As].bind(fio, thread_y())
    s[As].bind(fiio, thread_x())
    s[As].vectorize(fiii)

    _, b, x, h, y = s[Vs].leaf_iter_vars
    s[Vs].reorder(b, h, x, y)
    f = s[Vs].fuse(x, y)
    fo, fi = s[Vs].split(f, factor = ntx * nty * 4)
    fio, fii = s[Vs].split(fi, factor = ntx * 4)
    fiio, fiii = s[Vs].split(fii, factor = 4)
    s[Vs].bind(fio, thread_y())
    s[Vs].bind(fiio, thread_x())
    s[Vs].vectorize(fiii)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        V: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (l2ufw.get_fn(lens)(b))),
        A: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (l1ufw.get_fn(lens)(b) *
                                                                  l2ufw.get_fn(lens)(b))),
        O: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: l1ufw.get_fn(lens)(b))
    }

inputs = [[lens], [BS_VAR, V, A, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))

# _, V, A, O  = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     rounded64 = utils.ceilmult(length, 64)
#     rounded16 = utils.ceilmult(length, 16)
#     this_extent = rounded64 * NUM_HEADS * HEAD_SIZE
#     # print(length, rounded16, run_utils.stats(O[ctr:ctr + 1]))
#     print(length, O[ctr], O[ctr + 512], O[ctr + (length - 1)*512])
#     ctr += this_extent
