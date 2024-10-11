import numpy as np
import os
import argparse
import tvm
from tvm import tir, te
from tvm.te import RangeDimension as Dim
from tvm.tir import UninterpFun as Uf, UfWrapper as Ufw
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import utils
import run_utils

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--op', dest='op', nargs='?', default='mul')
parser.add_argument('--m', dest='m', default=1024, type=int)
parser.add_argument('--only-prep-code', dest='only_prep_code', default=None, type=str)
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--disable-assert', dest='disable_assert', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
args = parser.parse_args()

M = args.m
md = Dim('md')
nd = Dim('nd')
fd = Dim('fd')

def len_ufw(name, pad): return Ufw(name, "l", (pad, M), [md], [], lambda: lambda m: utils.ceilmult(m + 1, pad))
luf = len_ufw('s2k', 1).get_uf()

ls =  {
    0: Uf.from_constant('md', M, 'l'),
    1: luf,
}

loop_ufs=[ls[0], ls[1]]
width_ufs=loop_ufs
A1 = te.ragged_placeholder((M, M), [md, nd], loop_ufs, name='A1', width_ufs=width_ufs)
A2 = te.ragged_placeholder((M, M), [md, nd], loop_ufs, name='A2', width_ufs=width_ufs)

def body_fn(ds):
    if args.op == 'mul': return A1[ds[md], ds[nd]] * A2[ds[md], ds[nd]]
    else: return A1[ds[md], ds[nd]] + A2[ds[md], ds[nd]]
O = te.ragged_compute((M, M), [md, nd], loop_ufs, body_fn, name = 'O', width_uf_lists=[width_ufs])

f_ext = (M * (M + 1)) // 2
rmap = te.fuse_ragged_axis([A1, A2], O, md, nd, fd, f_ext)
A1 = rmap[A1.op].output(0)
A2 = rmap[A2.op].output(0)
O = rmap[O.op].output(0)

s = tvm.create_schedule([O.op])

A1l = s.cache_read(A1, "local", [O], vanilla=True)
A2l = s.cache_read(A2, "local", [O], vanilla=True)
Ol = s.cache_write(O, "local")

if M == 128: tile,ntx=1,64
elif M == 256: tile,ntx=2,64
elif M == 512: tile,ntx=4,64
else: tile,ntx=4,128

f = s[O].op.axis[0]
fo, fi = s[O].split(f, factor=ntx*tile)
fio, fii = s[O].split(fi, factor=tile)

s[O].bind(fo, tvm.thread_axis("blockIdx.x"))
s[O].bind(fio, tvm.thread_axis("threadIdx.x"))

s[Ol].unroll(s[Ol].leaf_iter_vars[0])

s[A1l].compute_at(s[O], fio)
s[A2l].compute_at(s[O], fio)
s[Ol].compute_at(s[O], fio)
s[A1l].vectorize(s[A1l].leaf_iter_vars[0])
s[A2l].vectorize(s[A2l].leaf_iter_vars[0])
s[O].vectorize(fii)

gen_prefix = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
_ = tvm.register_func(utils.get_tvm_callback_cuda_compile(256))
_ = tvm.register_func(
    utils.get_tvm_callback_cuda_postproc(args, os.path.realpath(__file__), fileprefix=gen_prefix))

bA1 = tvm.decl_buffer((f_ext,), name="bA1")
bA2 = tvm.decl_buffer((f_ext,), name="bA2")
bO = tvm.decl_buffer((f_ext,), name="bO")

binds={A1:bA1, A2:bA2, O:bO}
inputs = [[], [bA1, bA2, bO]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out = run_utils.lower_or_build(name, s, inputs, args, run_function=run_utils.run_trmm,
                               prep_code_mode='no_prep_code', binds=binds)
