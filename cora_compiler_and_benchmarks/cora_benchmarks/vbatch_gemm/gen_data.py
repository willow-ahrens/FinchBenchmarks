import random
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', dest='batch_size', default=32, type=int)
parser.add_argument('--out', dest='out', default=None, type=str)
parser.add_argument('--small', dest='small', default=False, action='store_true')
parser.add_argument('--small2', dest='small2', default=False, action='store_true')
args = parser.parse_args()

multiples = list(range(4, 12))
factor = 128

def ceil(a, f):
    return f * ((a + f - 1) // f)

with open(args.out, 'w') as outfile:
    for i in range(args.batch_size):
        if args.small:
            M = ceil(random.randrange(64, 384), 64)
            N = ceil(random.randrange(64, 384), 64)
            K = ceil(random.randrange(64, 384), 64)
        elif args.small2:
            M = random.randrange(16, 128)
            N = random.randrange(16, 128)
            K = ceil(random.randrange(16, 128), 16)
        else:
            M = factor * random.choice(multiples)
            N = factor * random.choice(multiples)
            K = factor * random.choice(multiples)
        print(M, N, K, file = outfile)
