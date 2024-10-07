import numpy as np
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
import utils
import run_utils

parser = run_utils.get_cmd_parser()
parser.add_argument('--nt', dest='nt', default=8, type=int)
parser.add_argument('--kt', dest='kt', default=4, type=int)
parser.add_argument('--sched', dest='sched', default=1, type=int)
parser.add_argument('--masked-mha', dest='masked_mha', default=False, action='store_true')
parser.add_argument('--no-hoist-loads', dest='no_hoist_loads', default=False, action='store_true')
args = parser.parse_args()
args.full_dense = True

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR
NUM_HEADS = 8
HEAD_SIZE = 64
TILE=64
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

ufw = Ufw('s', 'l', (1, MAX_LEN), [], [], lambda: lambda: MAX_LEN)
lufw = ufw
sufw = ufw

lbduf = Uf.from_constant('bd', BS_VAR, 'l')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufw.get_uf(),
    3: lufw.get_uf(),
    4: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    5: Uf.from_constant('qk', 3, 'l'),
}

loop_ufs=[ls[5], ls[0], ls[2], ls[1], ls[4]]
width_ufs = None # if args.dense_storage else [ls[5], ls[0], sufw.get_uf(), ls[1], ls[4]]
Q = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='Q', width_ufs=width_ufs)

loop_ufs=[ls[5], ls[0], ls[3], ls[1], ls[4]]
width_ufs = None # if args.dense_storage else [ls[5], ls[0], sufw.get_uf(), ls[1], ls[4]]
K = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s2, md, hd], loop_ufs,
                          name='K', width_ufs=width_ufs)

loop_ufs=[lbduf, ls[2], ls[1], ls[3]]
width_ufs = None # if args.dense_storage else [[ls[0], sufw.get_uf(), ls[1], sufw.get_uf()]]
k = tvm.reduce_axis((0, HEAD_SIZE), name = 'k')
S = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.sum(Q[0, ds[bd], ds[s1], ds[md], k] * K[1, ds[bd], ds[s2], ds[md], k],
                                         axis = k, dimensions=[hd]),
                      name = 'S', width_uf_lists=width_ufs)

def get_threshold(ds):
    if args.masked_mha:
        return ds[s1] + 1
    else:
        return lens[ds[bd]]

O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s1, md, s2], loop_ufs,
                      lambda ds: tvm.if_then_else(ds[s2] >= get_threshold(ds), -float('inf'), S[ds[bd], ds[s1], ds[md], ds[s2]]),
                      name = 'O', width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    thread_x = lambda: tvm.thread_axis('threadIdx.x')
    thread_y = lambda: tvm.thread_axis('threadIdx.y')
    block_x = lambda: tvm.thread_axis('blockIdx.x')
    block_y = lambda: tvm.thread_axis('blockIdx.y')

    if args.sched == 1:
        nt = 8

        Qs = s.cache_read(Q, 'shared', [S], layouts='dense')
        Ks = s.cache_read(K, 'shared', [S], layouts='dense')

        Ql = s.cache_read(Qs, 'local', [S], layouts='dense')
        Kl = s.cache_read(Ks, 'local', [S], layouts='dense')

        tile, ktile = 64, 8

        b, x, h, y = s[O].leaf_iter_vars[0:4]
        xo, xi = s[O].split(x, factor = tile)
        yo, yi = s[O].split(y, factor = tile)

        s[O].reorder(b, xo, yo, h, xi, yi)
        f1 = s[O].fuse(xo, yo)
        f2 = s[O].fuse(b, f1)
        s[O].bind(f2, block_y())
        s[O].bind(h, block_x())

        xio, xii = s[O].split(xi, factor = nt)
        yio, yii = s[O].split(yi, factor = nt)
        s[O].bind(xii, thread_y())
        s[O].bind(yii, thread_x())
        s[O].bind(yio, tvm.thread_axis('vthread', name='vth1'))
        s[O].bind(xio, tvm.thread_axis('vthread', name='vth2'))
        s[O].reorder(xio, yii, yio, xii)
        s[S].compute_at(s[O], xii)

        x, h, y, k = s[S].leaf_iter_vars[1:5]
        ko, ki = s[S].split(k, nparts = ktile)
        s[S].reorder(h, ko, ki, x, y)
        s[Qs].compute_at(s[S], ko)
        s[Ks].compute_at(s[S], ko)
        s[Ql].compute_at(s[S], ki)
        s[Kl].compute_at(s[S], ki)

        # x, h, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3], s[Ks].leaf_iter_vars[4]
        # yo, yi = s[Ks].split(y, factor = 4)
        # s[Ks].vectorize(yi)
        # fio, fii = s[Ks].split(x, factor = nt)
        # s[Ks].bind(fio, thread_y())
        # s[Ks].bind(fii, thread_x())

        # x, h, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3], s[Qs].leaf_iter_vars[4]
        # yo, yi = s[Qs].split(y, factor = 4)
        # s[Qs].vectorize(yi)
        # fio, fii = s[Qs].split(x, factor = nt)
        # s[Qs].bind(fio, thread_y())
        # s[Qs].bind(fii, thread_x())

        x, h, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3], s[Ks].leaf_iter_vars[4]
        s[Ks].reorder(h, y, x)
        s[Ks].bind(y, thread_y())
        fio, fii = s[Ks].split(x, factor = nt * 4)
        fiio, fiii = s[Ks].split(fii, factor = 4)
        s[Ks].bind(fiio, thread_x())
        s[Ks].vectorize(fiii)

        x, h, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3], s[Qs].leaf_iter_vars[4]
        s[Qs].reorder(h, y, x)
        s[Qs].bind(y, thread_y())
        fio, fii = s[Qs].split(x, factor = nt * 4)
        fiio, fiii = s[Qs].split(fii, factor = 4)
        s[Qs].bind(fiio, thread_x())
        s[Qs].vectorize(fiii)

        s.reorder_tensor_dimensions(Ks, 2, 3)
        s.reorder_tensor_dimensions(Ks, 3, 4)
        s.reorder_tensor_dimensions(Qs, 2, 3)
        s.reorder_tensor_dimensions(Qs, 3, 4)

        s[S].set_scope('local')
    else:
        nt = 8

        Qs = s.cache_read(Q, 'shared', [S], layouts='dense')
        Ks = s.cache_read(K, 'shared', [S], layouts='dense')

        Ql = s.cache_read(Qs, 'local', [S], layouts='dense')
        Kl = s.cache_read(Ks, 'local', [S], layouts='dense')

        tile, ktile = 32, 4

        b, x, h, y = s[O].leaf_iter_vars[0:4]
        xo, xi = s[O].split(x, factor = tile)
        yo, yi = s[O].split(y, factor = tile)

        s[O].reorder(b, xo, yo, h, xi, yi)
        f1 = s[O].fuse(xo, yo)
        f2 = s[O].fuse(b, f1)
        s[O].bind(f2, block_x())
        s[O].bind(h, block_y())

        xio, xii = s[O].split(xi, factor = nt)
        yio, yii = s[O].split(yi, factor = nt)
        s[O].bind(xii, thread_y())
        s[O].bind(yii, thread_x())
        s[O].bind(yio, tvm.thread_axis('vthread', name='vth1'))
        s[O].bind(xio, tvm.thread_axis('vthread', name='vth2'))
        s[O].reorder(xio, yii, yio, xii)
        s[S].compute_at(s[O], xii)

        x, h, y, k = s[S].leaf_iter_vars[1:5]
        ko, ki = s[S].split(k, nparts = ktile)
        s[S].reorder(h, ko, ki, x, y)
        s[Qs].compute_at(s[S], ko)
        s[Ks].compute_at(s[S], ko)
        s[Ql].compute_at(s[S], ki)
        s[Kl].compute_at(s[S], ki)

        x, h, y = s[Ks].leaf_iter_vars[2], s[Ks].leaf_iter_vars[3], s[Ks].leaf_iter_vars[4]
        s[Ks].reorder(h, y, x)
        f = s[Ks].fuse(x, y)
        fo, fi = s[Ks].split(f, factor = nt * nt * 4)
        fio, fii = s[Ks].split(fi, factor = nt * 4)
        fiio, fiii = s[Ks].split(fii, factor = 4)
        s[Ks].bind(fio, thread_y())
        s[Ks].bind(fiio, thread_x())
        s[Ks].vectorize(fiii)

        x, h, y = s[Qs].leaf_iter_vars[2], s[Qs].leaf_iter_vars[3], s[Qs].leaf_iter_vars[4]
        s[Qs].reorder(h, y, x)
        f = s[Qs].fuse(x, y)
        fo, fi = s[Qs].split(f, factor = nt * nt * 4)
        fio, fii = s[Qs].split(fi, factor = nt * 4)
        fiio, fiii = s[Qs].split(fii, factor = 4)
        s[Qs].bind(fio, thread_y())
        s[Qs].bind(fiio, thread_x())
        s[Qs].vectorize(fiii)

        s.reorder_tensor_dimensions(Ks, 2, 3)
        s.reorder_tensor_dimensions(Ks, 3, 4)
        s.reorder_tensor_dimensions(Qs, 2, 3)
        s.reorder_tensor_dimensions(Qs, 3, 4)

        s[S].set_scope('local')

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
        # Q: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (sufw.get_fn(lens)(b))),
        # K: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (sufw.get_fn(lens)(b))),
        # O: NUM_HEADS * run_utils.prefix_sum(len(lens),
                                            # lambda b: (sufw.get_fn(lens)(b) *
                                                       # sufw.get_fn(lens)(b)))
    }

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        hoist_loads=not args.no_hoist_loads, prep_code_mode='no_prep_code',
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR))


# _, Q, K, O = out[0:4]
# O = O.flatten()
# ctr = 0
# for length in batches[0]:
#     rounded = utils.ceilmult(length, TILE)
#     this_extent = rounded
#     this_storage_extent = rounded * rounded * NUM_HEADS
#     # print(rounded, np.mean(O[ctr:ctr+this_storage_extent]))
#     print(rounded, np.mean(O[ctr:ctr+length]))
#     ctr += this_storage_extent
