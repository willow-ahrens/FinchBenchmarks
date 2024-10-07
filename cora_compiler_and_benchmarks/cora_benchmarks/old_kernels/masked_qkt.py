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
parser.add_argument('--dataset', nargs='?', default='random_65_128')
parser.add_argument('--datadir', nargs='?', default='random')
args = parser.parse_args()

NUM_HEADS = 8
HEAD_SIZE = 64
TILE1=64
TILE2=64
RTILE=4
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), max(TILE1, TILE2))

lens = te.placeholder((args.batch_size,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

def len1_uf(name): return Uf(name, 'l', (TILE1, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], TILE1))
def len2_uf(name): return Uf(name, 'l', (TILE2, MAX_LEN), [s1], lambda s: utils.ceilmult(s + 1, TILE2))

luf1 = len1_uf('s1')
luf2 = len2_uf('s2')
ls =  {
    0: Uf.from_constant('bd', args.batch_size, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: luf1,
    3: luf1,
    4: Uf.from_constant('hd', HEAD_SIZE, 'l'),
}

loop_ufs=[ls[0], ls[1], ls[2], ls[4]]
width_ufs=[ls[0], ls[1], luf1, ls[4]]
Q = te.ragged_placeholder((args.batch_size, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s1, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[3], ls[4]]
width_ufs=[ls[0], ls[1], luf1, ls[4]]
K = te.ragged_placeholder((args.batch_size, NUM_HEADS, MAX_LEN, HEAD_SIZE), [bd, md, s2, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], luf1, luf2]
width_ufs=[loop_ufs]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
O = te.ragged_compute((args.batch_size, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[ds[bd], ds[md], ds[s1], k] * K[ds[bd], ds[md], ds[s2], k],
                                         axis = k, dimensions = [hd]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis("threadIdx.x")
thread_y = lambda: tvm.thread_axis("threadIdx.y")
block_x = lambda: tvm.thread_axis("blockIdx.x")
block_y = lambda: tvm.thread_axis("blockIdx.y")

Ol = s.cache_write(O, "local")
Qs = s.cache_read(Q, "shared", [Ol])
Ks = s.cache_read(K, "shared", [Ol])

Ql = s.cache_read(Qs, "local", [Ol])
Kl = s.cache_read(Ks, "local", [Ol])

b, h, x, y = s[O].leaf_iter_vars[0:4]
yo, yi = s[O].split(y, factor = 64)
xo, xi = s[O].split(x, factor = 64)

s[O].reorder(b, xo, yo, h, xi, yi)

f1 = s[O].fuse(xo, yo)
f = s[O].fuse(b, f1)
s[O].bind(f, block_x())
s[O].bind(h, block_y())
s[Qs].compute_at(s[O], h)
s[Ks].compute_at(s[O], h)

xio, xii = s[O].split(xi, factor = 16)
yio, yii = s[O].split(yi, factor = 16)
s[O].reorder(xii, yii, xio, yio)
s[O].bind(xii, thread_y())
s[O].bind(yii, thread_x())
s[O].bind(xio, tvm.thread_axis("vthread"))
s[O].bind(yio, tvm.thread_axis("vthread"))
s[Ol].compute_at(s[O], yio)

x, y, k = s[Ol].leaf_iter_vars[2], s[Ol].leaf_iter_vars[3], s[Ol].leaf_iter_vars[4]
s[Ol].reorder(k, x, y)
s[Ql].compute_at(s[Ol], k)
s[Kl].compute_at(s[Ol], k)

x, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3]
s[Qs].reorder(y, x)
f = s[Qs].fuse(x, y)
fo, fi = s[Qs].split(f, factor = 256)
fio, fii = s[Qs].split(fi, factor = 16)
s[Qs].bind(fio, thread_y())
s[Qs].bind(fii, thread_x())

x, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3]
s[Ks].reorder(y, x)
f = s[Ks].fuse(x, y)
fo, fi = s[Ks].split(f, factor = 256)
fio, fii = s[Ks].split(fi, factor = 16)
s[Ks].bind(fio, thread_y())
s[Ks].bind(fii, thread_x())

s.reorder_tensor_dimensions(Qs, 2, 3)
s.reorder_tensor_dimensions(Ks, 2, 3)

suffix = ""
gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0] + suffix
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=True):
    inputs = [[lens], [Q, K, O]]
    if args.debug_code:
        lowered = tvm.lower(s, inputs, args.target, simple_mode = True)
        print(lowered)
        # fadd, _ = tvm.build(s, inputs, args.target)
        # if args.target == 'cuda':
            # print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
        # else:
            # print('-----CPU code-----\n' + fadd.get_source())
    else:
        fadd, i_bufs = tvm.build(s, inputs, args.target)
        run_utils.run(fadd, i_bufs, inputs[1], args.batch_size, args.max_batches,
                      args.dataset, args.datadir, args.target, args.debug)
