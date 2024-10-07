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
parser.add_argument('--hfuse', dest='hfuse', default=False, action='store_true')
args = parser.parse_args()

BATCH_SIZE = te.var('bs')
MAX_LEN = run_utils.get_maxlen_padded(args.dataset)
assert MAX_LEN > 64
NUM_HEADS = 8
HEAD_SIZE = 64
red_tile = 8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')
hd = Dim('hd')
qk = Dim('qk')

def ubpw(name, pad): return Ufw(name, "l", (pad, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.ceilmult(lens[b], pad))
def lbw(name): return Ufw(name, "l", (0, MAX_LEN), [bd], [lens], lambda lens: lambda b: utils.floormult(lens[b], 64))
ubufw32 = ubpw('ub', 32)
ubufwrt = ubpw('ub', red_tile)
lbufw = lbw('lb')

def ub_pad(name, pad): return Uf(name, 'l', (pad, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], pad))
def lb(name): return Uf(name, 'l', (0, MAX_LEN), [bd], lambda b: utils.floormult(lens[b], 64))

qk_uf = Uf.from_constant('qk', 3, 'l')
bd_uf = Uf.from_constant('bd', BATCH_SIZE, 'l')
md_uf = Uf.from_constant('md', NUM_HEADS, 'l')
lb_uf = lbufw.get_uf()
ub_uf = ubufw32.get_uf()
hd_uf = Uf.from_constant('hd', HEAD_SIZE, 'l')
s1_uf = ubufwrt.get_uf()

loop_ufs=[bd_uf, ub_uf, md_uf, s1_uf]
width_ufs=loop_ufs
A = te.ragged_placeholder((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), [bd, s2, md, s1], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[qk_uf, bd_uf, s1_uf, md_uf, hd_uf]
width_ufs=loop_ufs
V = te.ragged_placeholder((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [qk, bd, s1, md, hd], loop_ufs,
                          name='V', width_ufs=width_ufs)

loop_ufs=[bd_uf, ub_uf, md_uf, hd_uf]
width_ufs=[loop_ufs]
O = te.ragged_compute((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), [bd, s2, md, hd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[s2], ds[md], rds['k']] *
                                              V(2, ds[bd], rds['k'], ds[md], ds[hd]),
                                              axis=rds['k'], dimensions=[s1]),
                      name = 'O', reduce_axis_ufs = [('k', ubpw('ub', 1).get_uf())],
                      width_uf_lists=width_ufs)

output_layout = O.op.output_layout(0)
s = tvm.create_schedule([O.op])

thread_x = lambda: tvm.thread_axis('threadIdx.x')
thread_y = lambda: tvm.thread_axis('threadIdx.y')
block_x = lambda: tvm.thread_axis('blockIdx.x')
block_y = lambda: tvm.thread_axis('blockIdx.y')
ntx = 8
nty = 8

def schedule_op(O, tile, suffix):
    Ol = s.cache_write(O, 'local')

    As = s.cache_read(A, 'shared', [Ol], suffix=suffix)
    Vs = s.cache_read(V, 'shared', [Ol], suffix=suffix)

    Al = s.cache_read(As, 'local', [Ol], suffix=suffix)
    Vl = s.cache_read(Vs, 'local', [Ol], suffix=suffix)

    b, x, h, y = s[O].leaf_iter_vars[0:4]
    xo, xi = s[O].split(x, factor = tile)

    s[O].reorder(b, xo, h, xi, y)
    f = s[O].fuse(b, xo)
    s[O].bind(f, block_y())
    s[O].bind(h, block_x())

    xio, xii = s[O].split(xi, factor = nty)
    yo, yi = s[O].split(y, factor = ntx)
    s[O].bind(xii, thread_y())
    s[O].bind(yi, thread_x())
    s[O].bind(xio, tvm.thread_axis('vthread', name='vth1'))
    s[O].bind(yo, tvm.thread_axis('vthread', name='vth2'))
    s[Ol].compute_at(s[O], yi)

    b, x, h, y, k = s[Ol].leaf_iter_vars
    s[Ol].reorder(b, h, k, x, y)
    ko, ki = s[Ol].split(k, factor = red_tile)
    s[As].compute_at(s[Ol], ko)
    s[Vs].compute_at(s[Ol], ko)
    s[Al].compute_at(s[Ol], ki)
    s[Vl].compute_at(s[Ol], ki)
    s[Ol].peel(ko)

    _, x, h, y = s[As].leaf_iter_vars
    s[As].reorder(h, x, y)
    f = s[As].fuse(x, y)
    fo, fi = s[As].split(f, factor = ntx * nty * 2)
    fio, fii = s[As].split(fi, factor = ntx * 2)
    fiio, fiii = s[As].split(fii, factor = 2)
    s[As].bind(fio, thread_y())
    s[As].bind(fiio, thread_x())
    if not args.debug_functions: s[As].vectorize(fiii)

    _, _, x, h, y = s[Vs].leaf_iter_vars
    s[Vs].reorder(h, x, y)
    f = s[Vs].fuse(x, y)
    fo, fi = s[Vs].split(f, factor = ntx * nty * 2)
    fio, fii = s[Vs].split(fi, factor = ntx * 2)
    fiio, fiii = s[Vs].split(fii, factor = 2)
    s[Vs].bind(fio, thread_y())
    s[Vs].bind(fiio, thread_x())
    if not args.debug_functions: s[Vs].vectorize(fiii)

G1, G2 = s.split_for_bin_packing([O], O, {O.op.axis[1]: lb_uf}, include_inputs=True)
O1, O2 = G1[0], G2[0]
if args.target == "cuda":
    schedule_op(O1, 64, '1')
    schedule_op(O2, 32, '2')

if args.hfuse:
    s.hfuse([(s[O1].op, s[O1].leaf_iter_vars[0]), (s[O2].op, s[O2].leaf_iter_vars[0])])

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        V: 3 * NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: (ubufwrt.get_fn(lens)(b))),
        A: NUM_HEADS * run_utils.prefix_sum(len(lens), lambda b: (ubufwrt.get_fn(lens)(b) *
                                                                  ubufw32.get_fn(lens)(b))),
        O: NUM_HEADS * HEAD_SIZE * run_utils.prefix_sum(len(lens), lambda b: ubufw32.get_fn(lens)(b))
    }

bO = tvm.tir.decl_buffer(output_layout, name="bO")
binds = {O1:bO, O2:bO}
if args.target == "cuda":
    inputs = [[lens], [BATCH_SIZE, V, A, bO]]
else:
    inputs = [[lens], [BATCH_SIZE, V, A, bO]]

name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, batches = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn, binds=binds,
                                        run_function=run_utils.get_bert_layer_run_fn(BATCH_SIZE))

_, V, A, O  = out
ctr = 0
O = O.flatten()
for length in batches[0]:
    rounded64 = utils.ceilmult(length, 32)
    this_extent = rounded64 * NUM_HEADS * HEAD_SIZE
    print(length, np.mean(O[ctr:ctr + this_extent]))
    ctr += this_extent
