import argparse
import tensorflow as tf
import numpy as np
import time
import timeit
import sys
sys.path.append("../")
import run_utils
import utils

tf.config.run_functions_eagerly(False)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--xla", help="Whether to run with TensorFlowXLA optimizations", action="store_true")
parser.add_argument("--print_tensorboard", help="Name of folder to output the tensorboard information")
parser.add_argument("--iterations", help="How many iterations to average for timing (default 5000)", type=int, default=5000)
parser.add_argument("--discard-iter", help="How many iterations to not time during warm up (default 1000)", type=int, default=1000)
args = parser.parse_args()

batch_size = 32
head_size = 64
num_heads = 8
ff_dim = 2048
model_dim = 512
max_len = 512

pre_w = tf.constant(np.random.random_sample((3*model_dim,model_dim)).astype(np.float32))
pre_b = tf.constant(np.random.random_sample((3*model_dim,)).astype(np.float32))
pst_w = tf.constant(np.random.random_sample((model_dim,model_dim)).astype(np.float32))
pst_b = tf.constant(np.random.random_sample((model_dim,)).astype(np.float32))
ff1_w = tf.constant(np.random.random_sample((ff_dim,model_dim)).astype(np.float32))
ff1_b = tf.constant(np.random.random_sample((ff_dim,)).astype(np.float32))
ff2_w = tf.constant(np.random.random_sample((model_dim,ff_dim)).astype(np.float32))

def layer_norm(input):
    mean = tf.math.reduce_mean(input, axis=2, keepdims=True)
    std = tf.math.reduce_std(input, axis=2, keepdims=True)
    return (input - mean) / std

def mha(input):
    pre_mm_out = tf.linalg.matmul(input, pre_w, transpose_b=True) + pre_b
    pre_out = tf.reshape(pre_mm_out, (batch_size,max_len,3,num_heads,head_size))
    q, k, v = tf.split(pre_out, 3, axis=2)
    q = tf.reshape(q, (batch_size,max_len,num_heads,head_size))
    k = tf.reshape(k, (batch_size,max_len,num_heads,head_size))
    v = tf.reshape(v, (batch_size,max_len,num_heads,head_size))
    q = tf.transpose(q, perm=[0,2,1,3])
    k = tf.transpose(k, perm=[0,2,1,3])
    v = tf.transpose(v, perm=[0,2,1,3])
    qkt_out = tf.matmul(q, k, transpose_b=True)
    attn_scores_out = tf.nn.softmax(qkt_out, axis=3)
    attn_out = tf.matmul(attn_scores_out, v)
    attn_out = tf.transpose(attn_out, perm=[0,2,1,3])
    attn_out = tf.reshape(attn_out, (batch_size,max_len,model_dim))
    pst_mm_out = tf.matmul(attn_out, pst_w, transpose_b=True)
    pst_out = layer_norm(pst_mm_out + pst_b + input)
    return pst_out

@tf.function(experimental_compile=args.xla)
def resnet50(input):
    mha_out = mha(input)
    ff1_mm_out = tf.linalg.matmul(mha_out, ff1_w, transpose_b=True)
    ff1_out = tf.nn.relu(ff1_mm_out + ff1_b)
    ff2_out = tf.linalg.matmul(ff1_out, ff2_w, transpose_b=True)
    ff_out = (ff2_out + input)
    return ff_out


# tf.profiler.experimental.start('log')
times = []
inputs = tf.constant(np.random.random_sample((batch_size,max_len,model_dim)).astype(np.float32))
for i in range(args.discard_iter + args.iterations):
    t0 = timeit.default_timer()
    resnet50(inputs)
    t1 = timeit.default_timer()
    times.append(t1 - t0)
# tf.profiler.experimental.stop()

total = 0
for i in range(args.discard_iter, len(times)):
    total += times[i]
avg = total / (args.iterations) * 1000.0
print("Average inference time of the last " + str(args.iterations) + " iterations: " + str(avg) + " ms")
