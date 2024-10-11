import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TVM_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/pad_fusion_layer.py'
TVM_LIB_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/gen_libs_pad_fusion.sh'
FTRANS_RUNNER = SCRIPT_DIR + '/../bert_layer/faster_transformer/run_encoder_sample.sh'
PYTHON = 'python3'

def generate_tvm_libs(dataset, pad_fusion):
    runner = TVM_LIB_RUNNER
    cmd = [runner, dataset, '1' if pad_fusion else '0']
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    print(out, err)

def get_cora_runner(pad_fusion):
    def run_cora(b_size, dataset, n_batch, err_file, args):
        log(args, ' Batch size %d' % (b_size))
        target = com.get_tvm_target(args.target)
        cmd = [PYTHON, TVM_EXE_RUNNER, '--target', target, '--batch-size', str(b_size),
               '--max-batches', str(n_batch), '--dataset', dataset]
        print(' '.join(cmd))
        if pad_fusion: cmd += ['--pad-fused']
        out, err = run_cmd(cmd)
        if err: print(err, file = err_file)
        return com.extract_times(out, 1)[0]
    return run_cora

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')

args = parser.parse_args()

b_sizes = [32, 64, 128]
targets = [args.target] if args.target else ['cuda']
# datasets = com.get_all_datasets() if args.dataset is None else [args.dataset]
datasets = ['race', 'mnli']

out_prefix = 'pad_fusion'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Target,Dataset,Batch Size,Unfused,Fused'
print(header, file = results_out)

unfused_runner = lambda b_sizes, *args: com.batchify(b_sizes, get_cora_runner(False), *args)
fused_runner = lambda b_sizes, *args: com.batchify(b_sizes, get_cora_runner(True), *args)

target = targets[0]
for dataset in datasets:
    exe_times = {}
    log(args, 'Unfused for ' + dataset)
    generate_tvm_libs(dataset, False)
    unfused_times = unfused_runner(b_sizes, dataset, args.max_batches, results_err, args)
    log(args, 'Fused for ' + dataset)
    generate_tvm_libs(dataset, True)
    fused_times = fused_runner(b_sizes, dataset, args.max_batches, results_err, args)
    print(fused_times)
    for b_size in b_sizes:
        out_str = '%s,%s,%d,%g,%g' % (target, dataset, b_size, unfused_times[b_size], fused_times[b_size])
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
