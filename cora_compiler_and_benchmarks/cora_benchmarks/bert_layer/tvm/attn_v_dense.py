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
parser.add_argument('--kt', dest='kt', default=8, type=int)
args = parser.parse_args()
args.full_dense = True

BS_VAR = te.var('bs')
BATCH_SIZE = BS_VAR + 1
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
NUM_HEADS = 8
HEAD_SIZE = 64

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

qk = Dim('qk')
bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')

ufw = Ufw('s', 'l', (1, MAX_LEN), [], [], lambda: lambda: MAX_LEN)
lufw1 = ufw
lufwp = ufw
sufwp = ufw

lbduf = Uf.from_constant('bd', BS_VAR, 'l')
ls = {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: lufwp.get_uf(),
    3: Uf.from_constant('hd', HEAD_SIZE, 'l'),
    4: Uf.from_constant('qk', 3, 'l'),
}

loop_ufs=[ls[0], ls[2], ls[1], ls[2]]
width_ufs=None
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s2, md, s1], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[4], ls[0], ls[2], ls[1], ls[3]]
width_ufs=None
V = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[lbduf, ls[2], ls[1], ls[3]]
width_ufs=None
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s2, md, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[s2], ds[md], rds['k']] *
                                              V[2, ds[bd], rds['k'], ds[md], ds[hd]],
                                              axis=rds['k'], dimensions=[s1]),
                      name = 'O', reduce_axis_ufs = [('k', lufw1.get_uf())],
                      width_uf_lists=width_ufs)

s = tvm.create_schedule([O.op])

if args.target == 'cuda':
    if args.dataset in ['mprc', 'cola']: nt, tile = 16, 32
    else: nt, tile = 8, 64

    thread_x = lambda: tvm.thread_axis('threadIdx.x')
    thread_y = lambda: tvm.thread_axis('threadIdx.y')
    block_x = lambda: tvm.thread_axis('blockIdx.x')
    block_y = lambda: tvm.thread_axis('blockIdx.y')

    Ol = s.cache_write(O, 'local')
    As = s.cache_read(A, 'shared', [Ol])
    Vs = s.cache_read(V, 'shared', [Ol])

    Al = s.cache_read(As, 'local', [Ol])
    Vl = s.cache_read(Vs, 'local', [Ol])


    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = tile)

    s[O].reorder(b, xo, h, xi, y)
    f = s[O].fuse(b, xo)
    s[O].bind(f, block_y())
    s[O].bind(h, block_x())

    xio, xii = s[O].split(xi, factor = nt)
    yo, yi = s[O].split(y, factor = nt)
    s[O].bind(xii, thread_y())
    s[O].bind(yi, thread_x())
    s[O].bind(xio, tvm.thread_axis('vthread', name='vth1'), no_unroll_vthread=args.debug_code)
    s[O].bind(yo, tvm.thread_axis('vthread', name='vth2'), no_unroll_vthread=args.debug_code)
    s[O].vectorize(yo)
    s[Ol].compute_at(s[O], yi)

    b, x, h, y, k = s[Ol].leaf_iter_vars
    s[Ol].reorder(b, h, k, x, y)
    ko, ki = s[Ol].split(k, factor = args.kt)
    s[As].compute_at(s[Ol], ko)
    s[Vs].compute_at(s[Ol], ko)
    s[Al].compute_at(s[Ol], ki)
    s[Vl].compute_at(s[Ol], ki)
    # s[Ol].peel(ko)

    if nt == 16: vtile = 1
    else: vtile = 4
    vtile=1
    _, x, h, y = s[As].leaf_iter_vars
    s[As].reorder(h, x, y)
    f = s[As].fuse(x, y)
    fo, fi = s[As].split(f, factor = nt * nt * vtile)
    fio, fii = s[As].split(fi, factor = nt * vtile)
    fiio, fiii = s[As].split(fii, factor = vtile)
    s[As].bind(fio, thread_x())
    s[As].bind(fiio, thread_y())
    s[As].vectorize(fiii)

    if nt == 16: vtile = 2
    else: vtile = 4
    vtile=1
    _, _, x, h, y = s[Vs].leaf_iter_vars
    s[Vs].reorder(h, x, y)
    s[Vs].bind(x, thread_x())
    fio, fii = s[Vs].split(y, factor = nt * vtile)
    fiio, fiii = s[Vs].split(fii, factor = vtile)
    s[Vs].bind(fiio, thread_y())
    s[Vs].vectorize(fiii)

    gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
    _ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
    _ = tvm.register_func(
        utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))


def size_fn(l_inputs):
    if args.no_raggedness or args.full_dense: return {}
    else:
        lens = l_inputs[0]
        return {
            V: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (sufwp.get_fn(lens)(b))),
            A: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (sufwp.get_fn(lens)(b) *
                                                                      sufwp.get_fn(lens)(b))),
            O: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: sufwp.get_fn(lens)(b))
        }

prep_code_mode = 'no_prep_code'
inputs = [[lens], [BS_VAR, V, A, O]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, pad_sum=64,
                                        run_function=run_utils.get_bert_layer_run_fn(BS_VAR),
                                        prep_code_mode=prep_code_mode)

# _, V, A, O  = out
# ctr = 0
# O = O.flatten()
# for length in batches[0]:
#     rounded64 = utils.ceilmult(length, 64)
#     this_extent = rounded64 * NUM_HEADS * HEAD_SIZE
#     print(length, run_utils.stats(O[ctr:ctr + this_extent]))
#     ctr += this_extent
