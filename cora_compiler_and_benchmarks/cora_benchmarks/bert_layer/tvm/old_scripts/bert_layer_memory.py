import os
import sys
import numpy as np
import time
import tvm
import argparse
import ast
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils
from common import OpShell as Op

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=200, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--bin-packed', dest='bin_packed', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random_384_512')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32)
NUM_HEADS = 8
HEAD_SIZE = 64
MODEL_DIM = NUM_HEADS * HEAD_SIZE
FF_DIM = 2048

dev_ctx = run_utils.get_ctx(args.target)
cpu_ctx = run_utils.get_ctx("llvm")

allocated_memory = 0
max_allocated_memory = -10000
ctr = 0
class Tensor():
    def __init__(self, size, ctx):
        global ctr
        self.name = ctr
        ctr += 1
        self.size = size
        self.ctx = ctx
        if 'gpu' in ctx.__repr__():
            # print('Alloc', self.name, size)
            global allocated_memory
            global max_allocated_memory
            allocated_memory += size
            max_allocated_memory = max(max_allocated_memory, allocated_memory)

    def free(self):
        if 'gpu' in self.ctx.__repr__():
            # print('Free', self.name, self.size)
            global allocated_memory
            allocated_memory -= self.size

def getnbytes(dt):
    if dt == 'float32': return 4
    elif dt == 'int32': return 4
    else:
        raise ValueError()

def create_tensor(shape, size, dtype, ctx):
    return Tensor(size * getnbytes(dtype), ctx)


ops = {
    'pre_linear': Op('pre_linear', 'pre_linear', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'qkt': Op('qkt', 'qkt_bin_packed' if args.bin_packed else 'qkt', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'softmax': Op('softmax', 'softmax', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'attn_v': Op('attn_v', 'attn_v_bin_packed' if args.bin_packed else 'attn_v', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'post_linear': Op('post_linear', 'post_linear', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'norm_add1': Op('norm_add1', 'norm_add', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'ff1': Op('ff1', 'ff1', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'ff2': Op('ff2', 'ff2', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
    'norm_add2': Op('norm_add2', 'norm_add', BATCH_SIZE, [], cpu_ctx, dev_ctx, alloc_op=create_tensor),
}

ops_order = [
    ops['pre_linear'],
    ops['qkt'],
    ops['softmax'],
    ops['attn_v'],
    ops['post_linear'],
    ops['norm_add1'],
    ops['ff1'],
    ops['ff2'],
    ops['norm_add2'],
]

# l_inputs: Allocate tensors
batches = run_utils.get_nlp_batches(args.batch_size, args.max_batches, args.dataset)
batches = run_utils.append_padded_sum(batches, 128)

pre_linear_in_w = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
pre_linear_in_b = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE,), "float32", dev_ctx, lw_args={})
post_linear_in_w = run_utils.create_tvm_array((NUM_HEADS * HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
post_linear_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
ff1_in_w = run_utils.create_tvm_array((MODEL_DIM, FF_DIM), "float32", dev_ctx, lw_args={})
ff1_in_b = run_utils.create_tvm_array((FF_DIM,), "float32", dev_ctx, lw_args={})
ff2_in_w = run_utils.create_tvm_array((FF_DIM, MODEL_DIM), "float32", dev_ctx, lw_args={})
ff2_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
norm_add1_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
norm_add1_in_g = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
norm_add2_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
norm_add2_in_g = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})

times = []
ctr = 0
for batch in batches:
    ctr += 1
    # print('BATCH', ctr)
    batch = np.sort(batch).astype('int32')
    batch_size_ = BATCH_SIZE + 1
    l_inputs = [tvm.nd.array(batch, cpu_ctx)]

    sum1 = run_utils.prefix_sum(batch_size_, lambda i: batch[i])
    sum16 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 16))
    sum32 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 32))
    sum64 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64))
    sum264 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64) * utils.ceilmult(batch[i], 64))

    # t_inputs: Allocate tensors
    pre_linear_in_qkv = create_tensor((batch_size_ * MAX_LEN, MODEL_DIM), sum1*MODEL_DIM, "float32", dev_ctx)
    pre_linear_out = create_tensor((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                   3*sum64*NUM_HEADS*HEAD_SIZE, "float32", dev_ctx)
    ops['pre_linear'].tensor_inputs = [pre_linear_in_qkv, pre_linear_in_w, pre_linear_in_b, pre_linear_out]
    # ops['pre_linear'].execute(l_inputs)

    qkt_in_q = pre_linear_out
    qkt_in_k = pre_linear_out
    qkt_out = create_tensor((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum264, "float32", dev_ctx)
    ops['qkt'].tensor_inputs = [qkt_in_q, qkt_in_k, qkt_out]
    # ops['qkt'].execute(l_inputs)

    softmax_in = qkt_out
    softmax_out = create_tensor((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum264, "float32", dev_ctx)
    ops['softmax'].tensor_inputs = [softmax_in, softmax_out]
    # ops['softmax'].execute(l_inputs)
    qkt_out.free()

    attn_v_in_attn = softmax_out
    attn_v_in_v = pre_linear_out
    attn_v_out = create_tensor((batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                               NUM_HEADS*HEAD_SIZE*sum64, "float32", dev_ctx)
    ops['attn_v'].tensor_inputs = [attn_v_in_v, attn_v_in_attn, attn_v_out]
    # ops['attn_v'].execute(l_inputs)
    pre_linear_out.free()
    softmax_out.free()

    post_linear_in_a = attn_v_out
    post_linear_out = create_tensor((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)
    ops['post_linear'].tensor_inputs = [post_linear_in_a, post_linear_in_w, post_linear_in_b, post_linear_out]
    # ops['post_linear'].execute(l_inputs)
    attn_v_out.free()

    # norm_add1_in_a1 = pre_linear_in_qkv.create_view((batch_size_, MAX_LEN, MODEL_DIM))
    norm_add1_in_a1 = pre_linear_in_qkv
    norm_add1_in_a2 = post_linear_out
    norm_add1_out = create_tensor((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)
    ops['norm_add1'].tensor_inputs = [norm_add1_in_a1, norm_add1_in_a2, norm_add1_in_b, norm_add1_in_g, norm_add1_out]
    # ops['norm_add1'].execute(l_inputs)
    pre_linear_in_qkv.free()
    post_linear_out.free()

    ff1_in_a = norm_add1_out
    ff1_out = create_tensor((batch_size_, MAX_LEN, FF_DIM), FF_DIM*sum1, "float32", dev_ctx)
    ops['ff1'].tensor_inputs = [ff1_in_a, ff1_in_w, ff1_in_b, ff1_out]
    # ops['ff1'].execute(l_inputs)

    ff2_in_a = ff1_out
    ff2_out = create_tensor((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)
    ops['ff2'].tensor_inputs = [ff2_in_a, ff2_in_w, ff2_in_b, ff2_out]
    # ops['ff2'].execute(l_inputs)
    ff1_out.free()

    norm_add2_in_a1 = norm_add1_out
    norm_add2_in_a2 = ff2_out
    norm_add2_out = create_tensor((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)
    ops['norm_add2'].tensor_inputs = [norm_add2_in_a1, norm_add2_in_a2, norm_add2_in_b, norm_add2_in_g, norm_add2_out]
    # ops['norm_add2'].execute(l_inputs)
    norm_add1_out.free()
    ff2_out.free()
    norm_add2_out.free()

print('MEM,%g' % (max_allocated_memory / (1024.0 * 1024.0)))
