import os
import argparse
import numpy as np
import time
import torch
import torch.nn.functional as f
from torch import Tensor
from torch import nn
import torch.utils.benchmark as benchmark
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../")
import run_utils
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='llvm')
parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
parser.add_argument('--max-batches', dest='max_batches', default=10, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--debug', dest='debug', default=False, action='store_true')
parser.add_argument('--dataset', nargs='?', default='random_384_512')
args = parser.parse_args()

def mean(l): return sum(l) / len(l)
np.random.seed(0)
VAL=0.1
def get_np_tensor(size, device, random, fill_value = None):
    if random:
        np_array = np.random.normal(size=size).astype('float32')
        return torch.randn(size, device = device, requires_grad = False, dtype = torch.float32)
    else:
        if fill_value == None: raise ValueError("No fill value provided " + str(fill_value))
        np_array = np.full(size, 0.1, 'float32').astype('float32')
    return torch.from_numpy(np_array).to(device)

class PreLinear(nn.Module):
    def __init__(self, device, max_len, batch_size, num_heads, head_size, model_size):
        super(PreLinear, self).__init__()
        self.pre_linear_w = get_np_tensor((3, num_heads, model_size, head_size), device, True)
        self.pre_linear_b = get_np_tensor((3, num_heads, 1, head_size), device, True)
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.model_size = model_size
        self.max_len = max_len

    def forward(self, inp):
        qkv = torch.matmul(inp, self.pre_linear_w)
        qkv += self.pre_linear_b
        qkv = qkv.view(3, self.num_heads, self.batch_size, self.max_len, self.head_size)
        q, k, v = torch.split(qkv, 1, 0)
        return q, k, v

class QKt(nn.Module):
    def __init__(self, device, max_len, batch_size, num_heads, head_size, model_size):
        super(QKt, self).__init__()

    def forward(self, q, k, attn_mask):
        attn = torch.matmul(q, k.permute(0, 1, 2, 4, 3))
        attn += attn_mask
        return attn

class Softmax(nn.Module):
    def __init__(self, device, max_len, batch_size, num_heads, head_size, model_size):
        super(Softmax, self).__init__()

    def forward(self, attn):
        return f.softmax(attn, dim = 4)

class AttnV(nn.Module):
    def __init__(self, device, max_len, batch_size, num_heads, head_size, model_size):
        super(AttnV, self).__init__()
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_size = head_size
        self.model_size = model_size
        self.max_len = max_len

    def forward(self, attn, v):
        attn = torch.reshape(torch.matmul(attn, v).permute(0, 2, 3, 1, 4), (self.batch_size, self.max_len, self.model_size))
        return attn

class PostLinear(nn.Module):
    def __init__(self, device, max_len, batch_size, num_heads, head_size, model_size):
        super(PostLinear, self).__init__()
        self.post_linear_w = get_np_tensor((model_size, model_size), device, True, VAL)
        self.post_linear_b = get_np_tensor((model_size,), device, True, VAL)

    def forward(self, attn):
        sa_out = torch.matmul(attn, self.post_linear_w)
        sa_out += self.post_linear_b
        return sa_out

num_heads = 8
head_size = 64
ff_size = 2048
model_size = num_heads * head_size
device = torch.device('cpu')
batch_size = args.batch_size

batches = run_utils.get_nlp_batches(batch_size, args.max_batches, args.dataset)

iters = 1 if args.mem or args.debug else 20

callable_to_profile = None
torch.set_num_threads(8)
op_times = {
    'pre_linear': [],
    'qkt': [],
    'softmax': [],
    'attn_v': [],
    'post_linear': [],
}
def run_for_batches():
    for batch in batches:
        max_len = int(np.amax(batch))

        attn_mask = np.full((batch_size, max_len, max_len), 0.0, dtype='float32')
        # for i in range(batch_size):
            # for j in range(max_len):
                # if j >= batch[i]:
                    # for k in range(0, max_len):
                        # attn_mask[i][j][k] = -float('inf')
                # else:
                    # for k in range(j + 1, max_len):
                        # attn_mask[i][j][k] = -float('inf')
        attn_mask = torch.from_numpy(attn_mask).to(device)
        encoder = MaskedMHA(device, max_len, batch_size, num_heads, head_size, model_size)
        traced_encoder = torch.jit.script(encoder)
        inp = get_np_tensor((args.batch_size * max_len, model_size), device, True)

        inp_t = get_np_tensor((args.batch_size * max_len, model_size), device, True)
        qkv = get_np_tensor((3, num_heads, batch_size, max_len, head_size), device, True)
        q, k, v = torch.split(qkv, 1, 0)
        attn = get_np_tensor((1, num_heads, batch_size, max_len, max_len), device, True)
        post_lin_in = get_np_tensor((1, num_heads, batch_size, max_len, head_size), device, True)

        ops = {
            'pre_linear': (PreLinear(device, max_len, batch_size, num_heads, head_size, model_size), [inp]),
            'qkt': (QKt(device, max_len, batch_size, num_heads, head_size, model_size), [q, k, attn_mask]),
            'softmax': (Softmax(device, max_len, batch_size, num_heads, head_size, model_size), [attn]),
            'attn_v': (AttnV(device, max_len, batch_size, num_heads, head_size, model_size), [attn, v]),
            'post_linear': (PostLinear(device, max_len, batch_size, num_heads, head_size, model_size), [post_lin_in]),
        }

        for k, v in ops.items():
            ops[k] = (torch.jit.script(v[0]), v[1])

        for k, v in ops:
            timer = benchmark.Timer(stmt='f(*inps)',
                                    globals={'inps': v[1], 'f': v[0]},
                                    num_threads=8)
            op_times[k].append(timer.timeit(iters).mean * 1000.0)

    return op_times

with torch.no_grad():
    op_times = run_for_batches()
    for ops in op_times:
        print('RESULTS', op, mean(op_times[op]), sep=',')
