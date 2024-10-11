import os
import argparse
import utils
import run_utils
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
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

BATCH_SIZE = args.batch_size
NUM_HEADS = 8
HEAD_SIZE = 64
TILE1 = 64
TILE2 = 64
TILE3 = 16
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), max(TILE1, TILE2, TILE3))

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len1_uf(name): return Uf(name, 'l', (TILE1, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], TILE1))
def len2_uf(name): return Uf(name, 'l', (TILE2, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], TILE2))
def len3_uf(name): return Uf(name, 'l', (TILE3, MAX_LEN), [s1], lambda s: utils.ceilmult(s + 1, TILE3))

luf3 = len3_uf('s2k')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: len1_uf('s1'),
    3: len2_uf('s2'),
    4: Uf.from_constant('hd', HEAD_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[4]]
width_ufs=loop_ufs
V = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s2, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2], ls[4]]
width_ufs=[loop_ufs]
O = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[md], ds[s1], rds['k']] *
                                              V(ds[bd], ds[md], rds['k'], ds[hd]),
                                              axis=rds['k'], dimensions = [s2]),
                      name = 'O', reduce_axis_ufs = [('k', luf3)],
                      width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Ol = s.cache_write(O, "local")
As = s.cache_read(A, "shared", [Ol])
Vs = s.cache_read(V, "shared", [Ol])

Al = s.cache_read(As, "local", [Ol])
Vl = s.cache_read(Vs, "local", [Ol])

b, h, x, y = s[O].leaf_iter_vars
xo, xi = s[O].split(x, factor = 64)

s[O].reorder(b, xo, h, y, xi)
f = s[O].fuse(b, xo)
s[O].bind(f, block_x())
s[O].bind(h, block_y())
s[Ol].compute_at(s[O], h)

xio, xii = s[O].split(xi, nparts = 8)
yo, yi = s[O].split(y, factor = 32)
s[O].reorder(xio, yi, yo, xii)
s[O].bind(xio, thread_y())
s[O].bind(yi, thread_x())
s[O].bind(yo, tvm.thread_axis("vthread"))
s[Ol].compute_at(s[O], yo)

x, y, k = s[Ol].leaf_iter_vars[2], s[Ol].leaf_iter_vars[3], s[Ol].leaf_iter_vars[4]
s[Ol].reorder(k, x, y)
ko, ki = s[Ol].split(k, factor = 16)
s[As].compute_at(s[Ol], ko)
s[Vs].compute_at(s[Ol], ko)
s[Al].compute_at(s[Ol], ki)
s[Vl].compute_at(s[Ol], ki)
s[Ol].peel(ko)

x, y = s[As].leaf_iter_vars[2], s[As].leaf_iter_vars[3]
f = s[As].fuse(x, y)
fo, fi = s[As].split(f, factor = 256)
fio, fii = s[As].split(fi, factor = 32)
s[As].bind(fio, thread_y())
s[As].bind(fii, thread_x())

x, y = s[Vs].leaf_iter_vars[2], s[Vs].leaf_iter_vars[3]
f = s[Vs].fuse(x, y)
fo, fi = s[Vs].split(f, factor = 256)
fio, fii = s[Vs].split(fi, factor = 32)
s[Vs].bind(fio, thread_y())
s[Vs].bind(fii, thread_x())

tvm_callback_cuda_compile = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))

with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=True):
    inputs = [[lens], [V, A, O]]
    if args.debug_code:
        # lowered = tvm.lower(s, inputs, simple_mode = True)
        # print(lowered)
        fadd, _ = tvm.build(s, inputs, args.target)
        if args.target == 'cuda':
            print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        else:
            print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd, i_bufs = tvm.build(s, inputs, args.target)
        # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
        run_utils.run(fadd, i_bufs, inputs[1], args.batch_size, args.max_batches,
                      args.dataset, args.datadir, args.target, args.debug)
