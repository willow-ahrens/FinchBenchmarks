import os
import time
import tvm
import argparse
import ast
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import utils
import run_utils
from common import Op

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

# t_inputs: Allocate tensors
pre_linear_in_qkv = run_utils.create_tvm_array((BATCH_SIZE * MAX_LEN, MODEL_DIM), "float32", dev_ctx, lw_args={})
pre_linear_in_w = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
pre_linear_in_b = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE,), "float32", dev_ctx, lw_args={})
pre_linear_out = run_utils.create_tvm_array((3, BATCH_SIZE, MAX_LEN, NUM_HEADS, MODEL_DIM), "float32", dev_ctx, lw_args={})

qkt_in_q = pre_linear_out
qkt_in_k = pre_linear_out
qkt_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), "float32", dev_ctx, lw_args={})

softmax_in = qkt_out
softmax_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, NUM_HEADS, MAX_LEN), "float32", dev_ctx, lw_args={})

attn_v_in_attn = softmax_out
attn_v_in_v = pre_linear_out
attn_v_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, NUM_HEADS, HEAD_SIZE), "float32", dev_ctx, lw_args={})

post_linear_in_a = attn_v_out
post_linear_in_w = run_utils.create_tvm_array((NUM_HEADS * HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
post_linear_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
# post_linear_out = run_utils.create_tvm_array((BATCH_SIZE * MAX_LEN, MODEL_DIM), "float32", dev_ctx, lw_args={})
post_linear_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, MODEL_DIM), "float32", dev_ctx, lw_args={})

norm_add1_in_a1 = pre_linear_in_qkv.create_view((BATCH_SIZE, MAX_LEN, MODEL_DIM))
# norm_add1_in_a2 = post_linear_out.create_view((BATCH_SIZE, MAX_LEN, MODEL_DIM))
norm_add1_in_a2 = post_linear_out
norm_add1_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, MODEL_DIM), "float32", dev_ctx, lw_args={})

ff1_in_a = norm_add1_out
ff1_in_w = run_utils.create_tvm_array((MODEL_DIM, FF_DIM), "float32", dev_ctx, lw_args={})
ff1_in_b = run_utils.create_tvm_array((FF_DIM,), "float32", dev_ctx, lw_args={})
ff1_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, FF_DIM), "float32", dev_ctx, lw_args={})

ff2_in_a = ff1_out
ff2_in_w = run_utils.create_tvm_array((FF_DIM, MODEL_DIM), "float32", dev_ctx, lw_args={})
ff2_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})
ff2_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, MODEL_DIM), "float32", dev_ctx, lw_args={})

norm_add2_in_a1 = norm_add1_out
norm_add2_in_a2 = ff2_out
norm_add2_out = run_utils.create_tvm_array((BATCH_SIZE, MAX_LEN, MODEL_DIM), "float32", dev_ctx, lw_args={})

ops = [
    Op('pre_linear', 'pre_linear', BATCH_SIZE,
       [pre_linear_in_qkv, pre_linear_in_w, pre_linear_in_b, pre_linear_out], cpu_ctx, dev_ctx),
    Op('qkt', 'qkt_bin_packed' if args.bin_packed else 'qkt', BATCH_SIZE, [qkt_in_q, qkt_in_k, qkt_out], cpu_ctx, dev_ctx),
    Op('softmax', 'softmax', BATCH_SIZE, [softmax_in, softmax_out], cpu_ctx, dev_ctx),
    Op('attn_v', 'attn_v_bin_packed' if args.bin_packed else 'attn_v', BATCH_SIZE,
       [attn_v_in_v, attn_v_in_attn, attn_v_out], cpu_ctx, dev_ctx),
    Op('post_linear', 'post_linear', BATCH_SIZE,
       [post_linear_in_a, post_linear_in_w, post_linear_in_b, post_linear_out], cpu_ctx, dev_ctx),
    Op('norm_add1', 'norm_add', BATCH_SIZE, [norm_add1_in_a1, norm_add1_in_a2, norm_add1_out], cpu_ctx, dev_ctx),
    Op('ff1', 'ff1', BATCH_SIZE, [ff1_in_a, ff1_in_w, ff1_in_b, ff1_out], cpu_ctx, dev_ctx),
    Op('ff2', 'ff2', BATCH_SIZE, [ff2_in_a, ff2_in_w, ff2_in_b, ff2_out], cpu_ctx, dev_ctx),
    Op('norm_add2', 'norm_add', BATCH_SIZE, [norm_add2_in_a1, norm_add2_in_a2, norm_add2_out], cpu_ctx, dev_ctx),
]

# l_inputs: Allocate tensors
batches = run_utils.get_nlp_batches(args.batch_size, args.max_batches, args.dataset)
batches = run_utils.add_padded_sum(batches, 128)

times = []
witers = 10
iters = 10
for batch in batches:
    sorted(batch)
    l_inputs = [tvm.nd.array(batch, cpu_ctx)]

    for i in range(witers):
        for op in ops: op.execute(l_inputs)
        dev_ctx.sync()

    start = time.perf_counter()
    for i in range(iters):
        for op in ops: op.execute(l_inputs)
        dev_ctx.sync()
    end = time.perf_counter()
    times.append(end - start)

total_time = sum(times)*1000.0
print(total_time / (len(batches) * iters))
