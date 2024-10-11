import os
import run_utils
import argparse
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

BATCH_SIZE = args.batch_size
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
NUM_HEADS = 8
scale = 1/8

lens = te.placeholder((BATCH_SIZE,), name = 'lens', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
s1 = Dim('s1')
s2 = Dim('s2')

def len_uf(name, padding): return Uf(name, 'l', (padding, MAX_LEN), [bd], lambda b: utils.ceilmult(lens[b], padding))

luf1 = len_uf('s1_1', 1)
luf32 = len_uf('s2_32', 32)
luf64 = len_uf('s64', 64)
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, 'l'),
    1: Uf.from_constant('md', NUM_HEADS, 'l'),
    2: luf1,
    3: luf32,
}

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
width_ufs=[ls[0], ls[1], luf64, luf64]
A = te.ragged_placeholder((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                          name='A', width_ufs=width_ufs)

loop_ufs=[ls[0], ls[1], ls[2]]
Amax = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN), [bd, md, s1], loop_ufs,
                         lambda ds, rds: tvm.max(A[ds[bd], ds[md], ds[s1], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Amax', reduce_axis_ufs = [('k', luf32)])

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
Aexp = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                         lambda ds: tvm.exp((A[ds[bd], ds[md], ds[s1], ds[s2]] -
                                             Amax[ds[bd], ds[md], ds[s1]]) * scale), name = 'Aexp')

loop_ufs=[ls[0], ls[1], ls[2]]
Asum = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN), [bd, md, s1], loop_ufs,
                         lambda ds, rds: tvm.sum(Aexp[ds[bd], ds[md], ds[s1], rds['k']], axis=rds['k'], dimensions=s2),
                         name = 'Asum', reduce_axis_ufs = [('k', luf32)])

loop_ufs=[ls[0], ls[1], ls[2], ls[3]]
O = te.ragged_compute((BATCH_SIZE, NUM_HEADS, MAX_LEN, MAX_LEN), [bd, md, s1, s2], loop_ufs,
                      lambda ds: Aexp[ds[bd], ds[md], ds[s1], ds[s2]] / Asum[ds[bd], ds[md], ds[s1]],
                      name = 'O', width_uf_lists=[width_ufs])

s = tvm.create_schedule([O.op])

thread_x = tvm.thread_axis("threadIdx.x")
thread_y = tvm.thread_axis("threadIdx.y")
block_x = tvm.thread_axis("blockIdx.x")
block_y = tvm.thread_axis("blockIdx.y")


ko, ki = s[Amax].split(s[Amax].op.reduce_axis[0], factor = 32)
Amax_rf = s.rfactor(Amax, ki, 1)

ko, ki = s[Asum].split(s[Asum].op.reduce_axis[0], factor = 32)
Asum_rf = s.rfactor(Asum, ki, 1)

b, h, s1, s2 = s[O].leaf_iter_vars
s[O].reorder(s1, h)
f = s[O].fuse(b, s1)
s[O].bind(f, block_x)
s[O].bind(h, thread_y)

xo, xi = s[O].split(s2, factor = 32)
s[O].bind(xi, thread_x)
s[Amax].bind(s[Amax].op.reduce_axis[0], thread_x)
s[Asum].bind(s[Asum].op.reduce_axis[0], thread_x)

s[Amax].compute_at(s[O], h)
s[Amax_rf].compute_at(s[Amax], s[Amax].leaf_iter_vars[3])
s[Asum].compute_at(s[O], h)
s[Asum_rf].compute_at(s[Asum], s[Asum].leaf_iter_vars[3])
s[Aexp].compute_inline()

s[Amax].set_scope('local')
s[Amax_rf].set_scope('local')
s[Asum].set_scope('local')
s[Asum_rf].set_scope('local')
s[Aexp].set_scope('local')

suffix = ""
gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0] + suffix
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

inputs = [[lens], [A, O]]
with tvm.build_config(prep_code_mode='with_prep_code', fill_in_function_bodies=True):
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
        # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/qkt.so')
        run_utils.run(fadd, i_bufs, inputs[1], args.batch_size, args.max_batches,
                      args.dataset, args.datadir, args.target, args.debug)
