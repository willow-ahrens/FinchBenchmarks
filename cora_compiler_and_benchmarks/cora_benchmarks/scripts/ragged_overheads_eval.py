import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PRE_LINEAR_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/pre_linear.py'
QKT_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/qkt.py'
SOFTMAX_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/softmax.py'
ATTN_V_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/attn_v_hoist_default.py'
POST_LINEAR_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/post_linear.py'

PRE_LINEAR_DENSE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/pre_linear_dense.py'
QKT_DENSE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/qkt_dense.py'
SOFTMAX_DENSE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/softmax_dense.py'
ATTN_V_DENSE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/attn_v_dense.py'
POST_LINEAR_DENSE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/post_linear_dense.py'
PYTHON = 'python3'

def get_runner(op, dense):
    if op == 'attn_v':
        if dense: return ATTN_V_DENSE_RUNNER
        else: return ATTN_V_RUNNER
    elif op == 'pre_linear':
        if dense: return PRE_LINEAR_DENSE_RUNNER
        else: return PRE_LINEAR_RUNNER
    elif op == 'post_linear':
        if dense: return POST_LINEAR_DENSE_RUNNER
        else: return POST_LINEAR_RUNNER
    elif op == 'qkt':
        if dense: return QKT_DENSE_RUNNER
        else: return QKT_RUNNER
    elif op == 'softmax':
        if dense: return SOFTMAX_DENSE_RUNNER
        else: return SOFTMAX_RUNNER

def run(op, target, b_sizes, n_batch, dense, vloops, vdims, hoist_loads, prelude):
    runner = get_runner(op, dense)

    cmd = ([PYTHON, runner, '--target', target, '--batch-sizes'] +
           [str(i) for i in b_sizes] +
           ['--max-batches', str(n_batch), '--dataset', 'random_512_512'])

    if dense: pass
    else:
        if not vdims: cmd += ['--dense-storage']
        if not hoist_loads: cmd += ['--no-hoist-loads']
        if prelude: cmd += ['--only-prep-code']

    print(' '.join(cmd))
    out, err = com.run_cmd(cmd)
    if err: print(err, file = results_err)
    return com.extract_time_batches(out)

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='cuda')
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

# ops = ['pre_linear', 'qkt', 'softmax', 'attn_v', 'post_linear']
# ops = ['attn_v', 'pre_linear']
ops = ['pre_linear']
# b_sizes = [32, 64, 128]
b_sizes = [64]

results_out, results_err = get_out_files(args, 'ragged_overheads', 'a' if args.append else 'w')
header = 'Op,Target,Batch Size,Dense,+vloops,+vdims,+LoadHoist'
print(header, file = results_out)

for op in ops:
    log(args, 'Running full dense %s' % (op))
    dense_times = run(op, args.target, b_sizes, args.max_batches, True, False, False, False, False)
    print(dense_times)

    log(args, 'Running +vloops %s' % (op))
    vloop_times = run(op, args.target, b_sizes, args.max_batches, False, True, False, False, False)
    print(vloop_times)
    vloop_preds = run(op, args.target, b_sizes, args.max_batches, False, True, False, False, True)
    print(vloop_preds)

    log(args, 'Running +vdims %s' % (op))
    vdim_times = run(op, args.target, b_sizes, args.max_batches, False, True, True, False, False)
    print(vdim_times)
    vdim_preds = run(op, args.target, b_sizes, args.max_batches, False, True, True, False, True)
    print(vdim_preds)

    log(args, 'Running +loadhoist %s' % (op))
    hoist_times = run(op, args.target, b_sizes, args.max_batches, False, True, True, True, False)
    print(hoist_times)

    for b_size in b_sizes:
        out_str = '%s,%s,%d,%g,%g,%g,%g' % (op, args.target, b_size,
                                            dense_times[b_size],
                                            vloop_times[b_size] - vloop_preds[b_size],
                                            vdim_times[b_size] - vdim_preds[b_size],
                                            hoist_times[b_size] - vdim_preds[b_size])
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
