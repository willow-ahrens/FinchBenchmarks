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
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=200, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random_384_512')
args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_HEADS = 8
HEAD_SIZE = 64
MODEL_DIM = NUM_HEADS * HEAD_SIZE
FF_DIM = 2048

def get_nbytes(dtype):
    if dtype == "float32": return 4
    else: raise ValueError()

class Allocator:
    def __init__(self):
        self.peak_memory_consumption = -10000
        self.current_memory_consumption = 0
        self.measure = False

    def alloc_ragged_tensor(self, shape, size, dtype):
        if self.measure:
            self.current_memory_consumption += size * get_nbytes(dtype)

    def alloc_dense_tensor(self, shape, dtype):
        size = 1
        for e in shape: size *= e
        if self.measure:
            self.current_memory_consumption += size * get_nbytes(dtype)

    def free_all(self):
        self.peak_memory_consumption = max(self.peak_memory_consumption,
                                           self.current_memory_consumption)
        self.current_memory_consumption = 0

    def start_profiling(self):
        self.measure = True

    def stop_profiling(self):
        self.measure = False

    def get_peak_memory_mbs(self):
        return self.peak_memory_consumption / (1024.0 * 1024.0)


# l_inputs: Allocate tensors
batches = run_utils.get_nlp_batches(args.batch_size, args.max_batches, args.dataset)
if args.dense_storage:
    batches = [[max(batch)] * args.batch_size for batch in batches]
else:
    batches = run_utils.append_padded_sum(batches, 64)

alloc = Allocator()

# pre_linear_in_w = alloc.alloc_dense_tensor((3, NUM_HEADS, HEAD_SIZE, MODEL_DIM), "float32")
# pre_linear_in_b = alloc.alloc_dense_tensor((3, NUM_HEADS, HEAD_SIZE,), "float32")
# post_linear_in_w = alloc.alloc_dense_tensor((NUM_HEADS * HEAD_SIZE, MODEL_DIM), "float32")
# post_linear_in_b = alloc.alloc_dense_tensor((MODEL_DIM,), "float32")
# norm_add1_in_b = alloc.alloc_dense_tensor((MODEL_DIM,), "float32")
# norm_add1_in_g = alloc.alloc_dense_tensor((MODEL_DIM,), "float32")
# norm_add2_in_b = alloc.alloc_dense_tensor((MODEL_DIM,), "float32")
# norm_add2_in_g = alloc.alloc_dense_tensor((MODEL_DIM,), "float32")
# ff1_in_w = alloc.alloc_dense_tensor((MODEL_DIM, FF_DIM), "float32")
# ff1_in_b = alloc.alloc_dense_tensor((FF_DIM,), "float32")
# ff2_in_w = alloc.alloc_dense_tensor((FF_DIM, MODEL_DIM), "float32")
# ff2_in_b = alloc.alloc_dense_tensor((MODEL_DIM,), "float32")

alloc.start_profiling()
for batch in batches:
    batch_size_ = len(batch)
    if args.dense_storage:
        sum1 = run_utils.prefix_sum(batch_size_, lambda i: batch[i])
        sum16 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 16))
        sum32 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 32))
        sum64 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64))
        sum264 = run_utils.prefix_sum(batch_size_, lambda i: utils.ceilmult(batch[i], 64) * utils.ceilmult(batch[i], 64))
    else:
        sum1 = run_utils.prefix_sum(batch_size_, lambda i: batch[i])
        sum16 = sum1
        sum32 = sum1
        sum64 = sum1
        sum264 = run_utils.prefix_sum(batch_size_, lambda i: batch[i] * batch[i])
    max_len = max(batch)

    inp = alloc.alloc_ragged_tensor((batch_size_ * max_len, MODEL_DIM), sum1*MODEL_DIM, "float32")
    pre_linear_out = alloc.alloc_ragged_tensor((3, batch_size_, max_len, NUM_HEADS, HEAD_SIZE),
                                         3*sum64*NUM_HEADS*HEAD_SIZE, "float32")
    qkt_out = alloc.alloc_ragged_tensor((batch_size_, max_len, NUM_HEADS, max_len), NUM_HEADS*sum264, "float32")
    softmax_out = alloc.alloc_ragged_tensor((batch_size_, max_len, NUM_HEADS, max_len), NUM_HEADS*sum264, "float32")
    attn_v_out = alloc.alloc_ragged_tensor((batch_size_, max_len, NUM_HEADS, HEAD_SIZE),
                                     NUM_HEADS*HEAD_SIZE*sum64, "float32")
    post_linear_out = alloc.alloc_ragged_tensor((batch_size_, max_len, MODEL_DIM), MODEL_DIM*sum1, "float32")
    layer_norm1_out = alloc.alloc_ragged_tensor((batch_size_, max_len, MODEL_DIM), MODEL_DIM*sum1, "float32")
    ff1_out = alloc.alloc_ragged_tensor((batch_size_, max_len, FF_DIM), FF_DIM*sum1, "float32")
    ff2_out = alloc.alloc_ragged_tensor((batch_size_, max_len, MODEL_DIM), MODEL_DIM*sum1, "float32")
    layer_norm2_out = alloc.alloc_ragged_tensor((batch_size_, max_len, MODEL_DIM), MODEL_DIM*sum1, "float32")

    alloc.free_all()
alloc.stop_profiling()
print("MEM", alloc.get_peak_memory_mbs(), sep=',')
