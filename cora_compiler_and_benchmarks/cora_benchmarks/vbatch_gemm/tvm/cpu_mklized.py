import numpy as np
import math
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

parser = run_utils.get_cmd_parser(no_options=True)
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--batch-sizes', dest='batch_sizes', nargs='+', default=[32], type=int)
parser.add_argument('--tile-size', dest='tile_size', default=128, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
parser.add_argument('--data-file', nargs='?', default='random')
args = parser.parse_args()

BATCH_SIZE = te.var('bs')

ms = tvm.decl_buffer((BATCH_SIZE,), name = 'ms', dtype = 'int32')
ns = tvm.decl_buffer((BATCH_SIZE,), name = 'ns', dtype = 'int32')
ks = tvm.decl_buffer((BATCH_SIZE,), name = 'ks', dtype = 'int32')

bd = Dim('bd')
md = Dim('md')
nd = Dim('nd')
kd = Dim('kd')
MIN_DIM, MAX_DIM = 4*128, 11*128

def f_mufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ms], lambda b: lambda b: args.tile_size * ms.vload(b))
def f_nufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ns], lambda b: lambda b: args.tile_size * ns.vload(b))
def f_kufw(name): return Ufw(name, "l", (MIN_DIM, MAX_DIM), [bd], [ks], lambda b: lambda b: args.tile_size * ks.vload(b))

mufw = f_mufw('m')
nufw = f_nufw('n')
kufw = f_kufw('k')
ls =  {
    0: Uf.from_constant('bd', BATCH_SIZE, "l"),
    1: mufw.get_uf(),
    2: nufw.get_uf(),
    3: kufw.get_uf(),
}

loop_ufs=[ls[0], ls[1], ls[3]]
A = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, kd], loop_ufs, name='A', width_ufs=None, dtype='float32')
loop_ufs=[ls[0], ls[3], ls[2]]
B = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, kd, nd], loop_ufs, name='B', width_ufs=None, dtype='float32')

loop_ufs=[ls[0], ls[1], ls[2]]
Op = te.ragged_placeholder((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs, name='Op', width_ufs=None, dtype='float32')

loop_ufs=[ls[0], ls[1], ls[2]]
S = te.ragged_compute((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs,
                      lambda ds, rds: tvm.sum(A[ds[bd], ds[md], rds['k']] * B[ds[bd], rds['k'], ds[nd]],
                                              axis=rds['k'], dimensions=[kd]),
                      name = 'S', reduce_axis_ufs = [('k', kufw.get_uf())], width_uf_lists=None)

alpha = 1
beta = 1
loop_ufs=[ls[0], ls[1], ls[2]]
O = te.ragged_compute((BATCH_SIZE, MAX_DIM, MAX_DIM), [bd, md, nd], loop_ufs,
                      lambda ds: alpha*S[ds[bd], ds[md], ds[nd]] + beta*Op[ds[bd], ds[md], ds[nd]],
                      name = 'O', width_uf_lists=None)

s = tvm.create_schedule([O.op])

tile = args.tile_size

def intrin_gemv(m, n, r):
    a = te.placeholder((m,r), name="a")
    b = te.placeholder((r,n), name="b")

    k = te.reduce_axis((0,r), name="k")
    c = te.ragged_compute((m, n), [md, nd], [Uf.from_constant('m', m, 'l'), Uf.from_constant('n', n, 'l')],
                          lambda ds: tvm.sum(a[ds[md], k] * b[k, ds[nd]], axis=k, dimensions=[kd]),
                          name = 'O', width_uf_lists=None)

    Ab = tvm.tir.decl_buffer(a.shape, a.dtype, name="A", offset_factor=1, strides=[MAX_DIM, 1])
    Bb = tvm.tir.decl_buffer(b.shape, b.dtype, name="B", offset_factor=1, strides=[MAX_DIM, 1])
    Cb = tvm.tir.decl_buffer(c.shape, c.dtype, name="C", offset_factor=1, strides=[tile, 1], scope='local')

    def intrin_func(ins, outs):
        aa, bb = ins
        cc = outs[0]

        def _body():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_packed("tvm.contrib.cblas.matmul_no_thread",
                                        aa,
                                        bb,
                                        cc,
                                        False,
                                        False,
                                        1.0,
                                        1.0))
            return ib.get()

        def _reduce_reset():
            ib = tvm.tir.ir_builder.create()
            ib.emit(tvm.tir.call_extern("int32", "gemv_reset", cc.access_ptr("w"), m, n))
            return ib.get()

        def _reduce_update():
            return _body()

        return _body(), _reduce_reset(), _reduce_update()

    return te.decl_tensor_intrin(c.op, intrin_func, binds={a: Ab, b: Bb, c: Cb})


def gemv_impl():
    cc_code = """
      extern "C" int gemv_reset(float *cc, int m, int n) {
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            cc[i * n + j] = 0.0;
          }
        }
        return 0;
      }
    """
    from tvm.contrib import util, clang

    temp = util.tempdir()
    ll_path = temp.relpath("temp.ll")
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code


gemv = intrin_gemv(tile, tile, tile)


prep_code_mode='with_prep_code'
if True:
    O_local = S
    b, m, n, k = tuple(O_local.op.axis) + tuple(O_local.op.reduce_axis)

    O_b, O_m, O_n = tuple(O.op.axis)
    O_m_o_o, O_m_i = s[O].split(O_m, factor=tile)
    O_m_i_o, O_m_i_i = s[O].split(O_m_i, factor=tile)
    O_n_o_o, O_n_i = s[O].split(O_n, factor=tile)
    O_n_i_o, O_n_i_i = s[O].split(O_n_i, factor=tile)

    s[O].reorder(O_b, O_m_o_o, O_n_o_o, O_m_i_o, O_n_i_o, O_m_i_i, O_n_i_i)

    fused = s[O].fuse(O_b, O_m_o_o)
    fused = s[O].fuse(O_n_o_o, fused)
    s[O].parallel(fused)

    s[O_local].compute_at(s[O], O_n_i_o)
    ko, ki = s[O_local].split(s[O_local].op.reduce_axis[0], factor=tile)
    s[O_local].reorder(s[O_local].leaf_iter_vars[0], ko, s[O_local].leaf_iter_vars[1],
                       s[O_local].leaf_iter_vars[2], ki)
    s[O_local].pragma(s[O_local].leaf_iter_vars[0], "import_llvm", gemv_impl())
    s[O_local].tensorize(s[O_local].leaf_iter_vars[2], gemv)

    s[S].set_scope('local')

def size_fn(l_inputs):
    lens = l_inputs[0]
    return {
        A: BATCH_SIZE * MAX_DIM * MAX_DIM,
        B: BATCH_SIZE * MAX_DIM * MAX_DIM,
        O: BATCH_SIZE * MAX_DIM * MAX_DIM,
    }

bO = tvm.tir.decl_buffer((BATCH_SIZE, MAX_DIM, MAX_DIM), name="bO")
binds = {Op: bO, O: bO}
if args.only_prep_code: prep_code_mode = 'only_prep_code'
inputs = [[ms, ns, ks], [BATCH_SIZE, A, B, bO]]
name = os.path.splitext(os.path.basename(os.path.realpath(__file__)))[0]
out, ms, ns, ks = run_utils.lower_or_build(name, s, inputs, args, size_fn=size_fn,
                                           run_function=run_utils.get_vbatch_gemm_run_fn(BATCH_SIZE),
                                           prep_code_mode=prep_code_mode, binds=binds)

# np.set_printoptions(threshold=sys.maxsize)
# _, A, B, O = out
# for i in range(len(ms)):
#     m = int(ms[0][i])
#     n = int(ns[0][i])
#     k = int(ks[0][i])
#     print(m, n, k, run_utils.stats(A[i,0:m,:]), run_utils.stats(B[i,:,0:n]), run_utils.stats(O[i,0:m,0:n]))
