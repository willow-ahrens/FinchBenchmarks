import math
import os
import argparse
import run_utils
import utils
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

MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 64)
NUM_HEADS = 8
HEAD_SIZE = 64
OUT_SIZE = 512

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
hd = Dim('hd')
od = Dim('od')
mdhd = Dim('mdhd')

def len64_uf(name): return Uf(name, 'l', (64, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], 64))
def len1_uf(name): return Uf(name, 'l', (1, MAX_LEN), [bd], lambda b: lens[b])

luf64 = len64_uf('s')
luf1 = len1_uf('s')
ls =  {
    0: Uf.from_constant('bd', args.batch_size, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: luf1,
    3: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    4: Uf.from_constant('od', OUT_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[ls[0], ls[1], luf64, ls[3]]
A = te.ragged_placeholder((args.batch_size, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], loop_ufs,
                          name='A', width_ufs=width_ufs)

W = te.placeholder((NUM_HEADS * HEAD_SIZE, OUT_SIZE), name='W')

loop_ufs=[ls[0], ls[2], ls[4]]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, NUM_HEADS * HEAD_SIZE), name = 'k')
O = te.ragged_compute((args.batch_size, MAX_LEN, OUT_SIZE), [bd, s1, od], loop_ufs,
                      lambda ds: tvm.sum(A[ds[bd], tvm.floordiv(k, HEAD_SIZE), ds[s1], tvm.floormod(k, HEAD_SIZE)] *
                                         W[k, ds[od]], axis=k, dimensions = [mdhd]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

tile = 128
rtile = 8
nt = tile // rtile
ks = utils.next_power_of_2((NUM_HEADS * HEAD_SIZE) / (6144 // tile))

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")
vthread = lambda: tvm.thread_axis("vthread")

Ol = s.cache_write(O, "local")
Ws = s.cache_read(W, "shared", [Ol], vanilla=True)
As = s.cache_read(A, "shared", [Ol])

Wl = s.cache_read(Ws, "local", [Ol], vanilla=True)
Al = s.cache_read(As, "local", [Ol])

b, l, h = s[O].leaf_iter_vars
y = s[O].fuse(b, l, padding = tile)
x = h
yo, yi = s[O].split(y, factor = tile)
xo, xi = s[O].split(x, factor = tile)
s[O].bind(yo, block_y())
s[O].bind(xo, block_x())

yio, yii = s[O].split(yi, factor = nt)
xio, xii = s[O].split(xi, factor = nt)
s[O].bind(xii, thread_x())
s[O].bind(yii, thread_y())
s[O].bind(xio, vthread(), no_unroll_vthread = True)
s[O].bind(yio, vthread(), no_unroll_vthread = True)
s[Ol].compute_at(s[O], xii)

b, x, y, k = s[Ol].leaf_iter_vars
s[Ol].reorder(k, x, y)
ko, ki = s[Ol].split(k, nparts = ks)
s[As].compute_at(s[Ol], ko)
s[Ws].compute_at(s[Ol], ko)
s[Al].compute_at(s[Ol], ki)
s[Wl].compute_at(s[Ol], ki)

b, h, l, i = s[As].leaf_iter_vars
s[As].reorder(h, b)
f = s[As].fuse(b, l)
f = s[As].fuse(f, i)
fo, fi = s[As].split(f, factor = nt * nt * 4)
fio, fii = s[As].split(fi, factor = nt * 4)
fiio, fiii = s[As].split(fii, factor = 4)
s[As].bind(fio, thread_y())
s[As].bind(fiio, thread_x())
s[As].vectorize(fiii)

s.reorder_tensor_dimensions(As, 0, 1)
s.fuse_tensor_dimensions(As, 1, 2)

s.fuse_tensor_dimensions(O, 0, 1)

x, y = s[Ws].leaf_iter_vars
f = s[Ws].fuse(x, y)
fo, fi = s[Ws].split(f, factor = nt * nt * 4)
fio, fii = s[Ws].split(fi, factor = nt * 4)
fiio, fiii = s[Ws].split(fii, factor = 4)
s[Ws].bind(fio, thread_y())
s[Ws].bind(fiio, thread_x())
s[Ws].vectorize(fiii)

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

bO = tvm.decl_buffer([args.batch_size * MAX_LEN, OUT_SIZE], name = "bA")
inputs = [[lens], [A, W, bO]]
with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=True):
    if args.debug_code:
        lowered = tvm.lower(s, inputs, args.target, simple_mode = True, binds = {O: bO})
        print(lowered)
        # fadd, _ = tvm.build(s, inputs, args.target, binds = {O: bO})
        # if args.target == 'cuda':
            # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        # else:
            # print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd, i_bufs = tvm.build(s, inputs, args.target, binds = {O: bO})
        # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
        run_utils.run(fadd, i_bufs, inputs[1], args.batch_size, args.max_batches,
                      args.dataset, args.datadir, args.target, args.debug)
