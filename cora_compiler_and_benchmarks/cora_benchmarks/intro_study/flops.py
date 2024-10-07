import os
import sys
import numpy as np
import time
import tvm
import argparse
import ast
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")
import run_utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--out-file', dest='out_file', type=str)
args = parser.parse_args()

def flops_for_dataset_batch(dataset, batch_size, max_batches):
    batches = run_utils.get_nlp_batches(batch_size, max_batches, dataset)

    MAX_LEN = utils.ceilmult(run_utils.get_dataset_max_len(dataset), 32)
    NUM_HEADS = 8
    HEAD_SIZE = 64
    MODEL_DIM = NUM_HEADS * HEAD_SIZE
    FF_DIM = 2048
    QKV_NUM = 3

    giga = 1e9
    total_dense_gflops = 0.0
    total_real_gflops = 0.0
    total_ragged_gflops = 0.0
    for batch in batches:
        if len(batch) != batch_size:
            continue

        batch_max_len = np.amax(batch)
        padded_batch = run_utils.append_padded_sum([batch], 64)[0]

        sum1 = run_utils.prefix_sum(batch_size, lambda i: batch[i])
        sum2 = run_utils.prefix_sum(batch_size, lambda i: batch[i] * batch[i])

        psum1 = run_utils.prefix_sum(batch_size + 1, lambda i: padded_batch[i])
        psum2 = run_utils.prefix_sum(batch_size + 1, lambda i: padded_batch[i] * padded_batch[i])
        psum264 = run_utils.prefix_sum(batch_size,
                                       lambda i: utils.ceilmult(padded_batch[i], 64) * utils.ceilmult(padded_batch[i], 64))
        psum21664 = run_utils.prefix_sum(batch_size,
                                         lambda i: utils.ceilmult(padded_batch[i], 16) * utils.ceilmult(padded_batch[i], 64))
        psum2132 = run_utils.prefix_sum(batch_size,
                                        lambda i: padded_batch[i] * utils.ceilmult(padded_batch[i], 32))

        # print(sum2, psum264, psum2132, psum21664, batch)
        def get_pre_linear_flops():
            dense_flops = (QKV_NUM * batch_size * batch_max_len * NUM_HEADS * HEAD_SIZE * MODEL_DIM + # MM
                           QKV_NUM * batch_size * batch_max_len * NUM_HEADS * HEAD_SIZE)  # Bias add
            real_flops = (QKV_NUM * psum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM +
                          QKV_NUM * psum1 * NUM_HEADS * HEAD_SIZE)
            ragged_flops = (QKV_NUM * sum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM +
                            QKV_NUM * sum1 * NUM_HEADS * HEAD_SIZE)
            return dense_flops, real_flops, ragged_flops

        def get_qkt_flops():
            dense_flops = batch_size * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE
            real_flops = NUM_HEADS * psum264 * HEAD_SIZE
            ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE
            return dense_flops, real_flops, ragged_flops

        def get_softmax_flops():
            dense_flops = batch_size * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE * 5 # (max + max_sub + exp + sum + exp_div)
            real_flops = NUM_HEADS * psum2132 * HEAD_SIZE * 5
            ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE * 5
            return dense_flops, real_flops, ragged_flops

        def get_attn_v_flops():
            dense_flops = batch_size * NUM_HEADS * batch_max_len * batch_max_len * HEAD_SIZE
            real_flops = NUM_HEADS * psum21664 * HEAD_SIZE
            ragged_flops = NUM_HEADS * sum2 * HEAD_SIZE
            return dense_flops, real_flops, ragged_flops

        def get_post_linear_flops():
            dense_flops = (batch_size * batch_max_len * NUM_HEADS * HEAD_SIZE * MODEL_DIM + # MM
                           batch_size * batch_max_len * MODEL_DIM)                          # Bias add
            real_flops = (psum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM +
                          psum1 * MODEL_DIM)
            ragged_flops = (sum1 * NUM_HEADS * HEAD_SIZE * MODEL_DIM +
                            sum1 * MODEL_DIM)
            return dense_flops, real_flops, ragged_flops

        def get_norm_add1_flops():
            dense_flops = batch_size * batch_max_len * MODEL_DIM * 4 # (residual_add + mean + std + div)
            real_flops = psum1 * MODEL_DIM * 4
            ragged_flops = sum1 * MODEL_DIM * 4
            return dense_flops, real_flops, ragged_flops

        def get_ff1_flops():
            dense_flops = (batch_size * batch_max_len * FF_DIM * MODEL_DIM + # MM
                           batch_size * batch_max_len * FF_DIM +             # Bias add
                           batch_size * batch_max_len * FF_DIM)              # Relu
            real_flops = (psum1 * FF_DIM * MODEL_DIM + # MM
                          psum1 * FF_DIM +             # Bias add
                          psum1 * FF_DIM)              # Relu
            ragged_flops = (sum1 * FF_DIM * MODEL_DIM + # MM
                            sum1 * FF_DIM +             # Bias add
                            sum1 * FF_DIM)              # Relu
            return dense_flops, real_flops, ragged_flops

        def get_ff2_flops():
            dense_flops = batch_size * batch_max_len * FF_DIM * MODEL_DIM              # Relu
            real_flops = psum1 * FF_DIM * MODEL_DIM
            ragged_flops = sum1 * FF_DIM * MODEL_DIM
            return dense_flops, real_flops, ragged_flops

        def get_norm_add2_flops():
            dense_flops = batch_size * batch_max_len * MODEL_DIM * 4 # (residual_add + mean + std + div)
            real_flops = psum1 * MODEL_DIM * 4
            ragged_flops = sum1 * MODEL_DIM * 4
            return dense_flops, real_flops, ragged_flops

        flops_ops = {
            'pre_linear': get_pre_linear_flops(),
            'qkt': get_qkt_flops(),
            'softmax': get_softmax_flops(),
            'attn_v': get_attn_v_flops(),
            'post_linear': get_post_linear_flops(),
            'norm_add1': get_norm_add1_flops(),
            'ff1': get_ff1_flops(),
            'ff2': get_ff2_flops(),
            'norm_add2': get_norm_add2_flops()
        }

        for name, op in flops_ops.items():
            total_dense_gflops += op[0] / giga
            total_real_gflops += op[1] / giga
            total_ragged_gflops += op[2] / giga

            # if op[1] <= op[2]:
                # print('SAME', batch_size, dataset, name, op)

    total_dense_gflops /= len(batches)
    total_real_gflops /= len(batches)
    total_ragged_gflops /= len(batches)
    return total_dense_gflops, total_real_gflops, total_ragged_gflops


with open(args.out_file, 'w') as out_file:
    print('Dataset', 'Batch Size', 'Dense Flops', 'Real Flops', 'Ragged Flops', sep=',', file=out_file)
    for dataset in run_utils.DATASETS:
        for i in range(8):
            batch_size = 1 << i
            dense, real, ragged = flops_for_dataset_batch(dataset, batch_size, args.max_batches)
            print(dataset, i, dense, real, ragged, sep=',', file=out_file)
