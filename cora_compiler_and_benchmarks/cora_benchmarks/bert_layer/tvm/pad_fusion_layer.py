import gc
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
from common import Op

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=200, type=int)
parser.add_argument('--witers', dest='witers', default=100, type=int)
parser.add_argument('--iters', dest='iters', default=200, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--pad-fused', dest='pad_fused', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random_384_512')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
MAX_LEN = max(64, utils.ceilmult(run_utils.get_dataset_max_len(args.dataset), 32))
NUM_HEADS = 8
HEAD_SIZE = 64
MODEL_DIM = NUM_HEADS * HEAD_SIZE

dev_ctx = run_utils.get_ctx(args.target)
cpu_ctx = run_utils.get_ctx("llvm")


if args.pad_fused:
    ops = {
        'pre_linear': Op('pre_linear', 'pre_linear', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'qkt': Op('qkt', 'qkt', BATCH_SIZE, [], cpu_ctx, dev_ctx, variants=[1]),
        'softmax': Op('softmax', 'softmax', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'attn_v': Op('attn_v', 'attn_v', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'post_linear': Op('post_linear', 'post_linear', BATCH_SIZE, [], cpu_ctx, dev_ctx),
    }
    ops_order = [
        ops['pre_linear'],
        ops['qkt'],
        ops['softmax'],
        ops['attn_v'],
        ops['post_linear'],
    ]
else:
    ops = {
        'pre_linear': Op('pre_linear', 'pre_linear', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'add_pad64': Op('add_pad64', 'padding_64to1_add', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'qkt': Op('qkt', 'qkt', BATCH_SIZE, [], cpu_ctx, dev_ctx, variants=[1]),
        'change_pad32': Op('change_pad32', 'padding_32to64_remove', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'softmax': Op('softmax', 'softmax', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'change_pad64': Op('change_pad64', 'padding_32to64_add', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'attn_v': Op('attn_v', 'attn_v', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'rem_pad64': Op('rem_pad64', 'padding_64to1_remove', BATCH_SIZE, [], cpu_ctx, dev_ctx),
        'post_linear': Op('post_linear', 'post_linear', BATCH_SIZE, [], cpu_ctx, dev_ctx),
    }
    ops_order = [
        ops['pre_linear'],
        ops['add_pad64'],
        ops['qkt'],
        ops['change_pad32'],
        ops['softmax'],
        ops['change_pad64'],
        ops['attn_v'],
        ops['rem_pad64'],
        ops['post_linear'],
    ]

# l_inputs: Allocate tensors
batches = run_utils.get_nlp_batches(args.batch_size, args.max_batches, args.dataset)
batches = run_utils.reverse_sort_batches(batches)
batches = run_utils.append_padded_sum(batches, 64)

pre_linear_in_w = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
pre_linear_in_b = run_utils.create_tvm_array((3, NUM_HEADS, HEAD_SIZE,), "float32", dev_ctx, lw_args={})
post_linear_in_w = run_utils.create_tvm_array((NUM_HEADS * HEAD_SIZE, MODEL_DIM), "float32", dev_ctx, lw_args={})
post_linear_in_b = run_utils.create_tvm_array((MODEL_DIM,), "float32", dev_ctx, lw_args={})

times = []
time_dict = {}
batch_size_ = BATCH_SIZE + 1

for batch in batches:
    sum1 = run_utils.prefix_sum(batch_size_, lambda i: batch[i])
    sum16 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 16))
    sum32 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 32))
    sum64 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64))
    sum232 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 32) * utils.ceilmult(batch[i], 32))
    sum264 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64) * utils.ceilmult(batch[i], 64))

    if args.pad_fused:
        pre_linear_in_qkv = run_utils.create_ragged_array((batch_size_ * MAX_LEN, MODEL_DIM), sum1*MODEL_DIM, "float32", dev_ctx)
        pre_linear_out = run_utils.create_ragged_array((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                       3*sum64*NUM_HEADS*HEAD_SIZE, "float32", dev_ctx)

        qkt_in_q = pre_linear_out
        qkt_in_k = pre_linear_out
        qkt_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum264, "float32", dev_ctx)

        softmax_in = qkt_out
        softmax_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum264, "float32", dev_ctx)

        attn_v_in_attn = softmax_out
        attn_v_in_v = pre_linear_out
        attn_v_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                   NUM_HEADS*HEAD_SIZE*sum64, "float32", dev_ctx)

        post_linear_in_a = attn_v_out
        post_linear_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)


        ops['pre_linear'].tensor_inputs = [pre_linear_in_qkv, pre_linear_in_w, pre_linear_in_b, pre_linear_out]
        ops['qkt'].tensor_inputs = [qkt_in_q, qkt_in_k, qkt_out]
        ops['softmax'].tensor_inputs = [softmax_in, softmax_out]
        ops['attn_v'].tensor_inputs = [attn_v_in_v, attn_v_in_attn, attn_v_out]
        ops['post_linear'].tensor_inputs = [post_linear_in_a, post_linear_in_w, post_linear_in_b, post_linear_out]

    else:
        pre_linear_in_qkv = run_utils.create_ragged_array((batch_size_ * MAX_LEN, MODEL_DIM), sum1*MODEL_DIM, "float32", dev_ctx)
        pre_linear_out = run_utils.create_ragged_array((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                       3*sum1*NUM_HEADS*HEAD_SIZE, "float32", dev_ctx)

        add_pad64_in_a = pre_linear_out
        add_pad64_out = run_utils.create_ragged_array((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                       3*sum64*NUM_HEADS*HEAD_SIZE, "float32", dev_ctx)

        qkt_in_q = add_pad64_out
        qkt_in_k = add_pad64_out
        qkt_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum264, "float32", dev_ctx)

        change_pad32_in_a = qkt_out
        change_pad32_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN),
                                                         NUM_HEADS*sum232, "float32", dev_ctx)

        softmax_in = change_pad32_out
        softmax_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN), NUM_HEADS*sum232, "float32", dev_ctx)

        change_pad64_in_a = softmax_out
        change_pad64_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, MAX_LEN),
                                                         NUM_HEADS*sum264, "float32", dev_ctx)

        attn_v_in_attn = change_pad64_out
        attn_v_in_v = add_pad64_out
        attn_v_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                   NUM_HEADS*HEAD_SIZE*sum64, "float32", dev_ctx)

        rem_pad64_in_a = attn_v_out
        rem_pad64_out = run_utils.create_ragged_array((3, batch_size_, MAX_LEN, NUM_HEADS, HEAD_SIZE),
                                                      3*sum1*NUM_HEADS*HEAD_SIZE, "float32", dev_ctx)

        post_linear_in_a = rem_pad64_out
        post_linear_out = run_utils.create_ragged_array((batch_size_, MAX_LEN, MODEL_DIM), MODEL_DIM*sum1, "float32", dev_ctx)


        ops['pre_linear'].tensor_inputs = [pre_linear_in_qkv, pre_linear_in_w, pre_linear_in_b, pre_linear_out]
        ops['add_pad64'].tensor_inputs = [add_pad64_in_a, add_pad64_out]
        ops['qkt'].tensor_inputs = [qkt_in_q, qkt_in_k, qkt_out]
        ops['change_pad32'].tensor_inputs = [change_pad32_in_a, change_pad32_out]
        ops['softmax'].tensor_inputs = [softmax_in, softmax_out]
        ops['change_pad64'].tensor_inputs = [change_pad64_in_a, change_pad64_out]
        ops['attn_v'].tensor_inputs = [attn_v_in_v, attn_v_in_attn, attn_v_out]
        ops['rem_pad64'].tensor_inputs = [rem_pad64_in_a, rem_pad64_out]
        ops['post_linear'].tensor_inputs = [post_linear_in_a, post_linear_in_w, post_linear_in_b, post_linear_out]


    l_inputs = [tvm.nd.array(batch, cpu_ctx)]

    # for op in ops_order: op.set_inputs_and_variant(l_inputs, 0)
    # for i in range(args.witers):
        # for op in ops_order:
            # op.execute()
            # dev_ctx.sync()







    for op in ops_order: op.set_inputs_and_variant(l_inputs, 0)
    for i in range(args.witers):
        for op in ops_order: op.execute()
    dev_ctx.sync()

    start = time.perf_counter()
    for i in range(args.iters):
        for op in ops_order: op.execute()
    dev_ctx.sync()
    end = time.perf_counter()
    times.append((end - start) / args.iters)

    for op in ops_order: op.reset()
    gc.collect()

total_time = sum(times)*1000.0
print('RESULTS', total_time / (len(batches)), sep=',')
