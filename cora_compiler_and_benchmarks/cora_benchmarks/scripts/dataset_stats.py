import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PYTORCH_RUNNER = SCRIPT_DIR + '/../bert_layer/pytorch/layer.py'
TVM_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/masked_mha.py'
TVM_MEM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/bert_layer_memory.py'
TVM_LIB_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/gen_libs.sh'
FTRANS_RUNNER = SCRIPT_DIR + '/../bert_layer/faster_transformer/run_encoder_sample.sh'
PYTHON = 'python3'

def generate_tvm_libs(dataset, args):
    cmd = [TVM_LIB_RUNNER, dataset,
           '1' if args.bin_packed else '0',
           '1' if args.masked_mha else '0',
           '1' if args.prep_overhead else '0']
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    print(out, err)

def run_pytorch(b_size, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    cmd = [PYTHON, PYTORCH_RUNNER, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
           '--max-batches', str(n_batch), '--dataset', dataset]
    if args.mem: cmd += ['--mem']

    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)

    if args.mem: return com.extract_mem(out)
    else: return com.extract_times(out, 1)[0]

def get_ftrans_runner(no_pad):
    def run_ftrans(b_size, dataset, n_batch, err_file, args):
        log(args, ' Batch size %d' % (b_size))
        num_layers = 25
        runner = FTRANS_RUNNER
        cmd = [runner, com.get_dataset_file(dataset), str(b_size), str(n_batch),
               str(com.get_dataset_max_len(dataset)), str(num_layers), '1' if no_pad else '0']

        # print(' '.join(cmd))
        out, err = run_cmd(cmd)
        # print(out)
        if err: print(err, file = err_file)

        if args.mem: return com.extract_mem(out)
        else: return com.extract_times(out, 1)[0] / num_layers
    return run_ftrans

def run_tvm(b_size, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    runner = TVM_MEM_RUNNER if args.mem else TVM_EXE_RUNNER

    cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
           '--max-batches', str(n_batch), '--dataset', dataset]
    if args.bin_packed: cmd += ['--bin-packed']
    if args.masked_mha: cmd += ['--masked-mha']
    print(' '.join(cmd))
    out, err = '', ''
    out, err = run_cmd(cmd)
    print(out)
    if err: print(err, file = err_file)

    if args.mem: return com.extract_mem(out)
    else: return com.extract_times(out, 1)[0]

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--batch-size', dest='batch_size', default=128, type=int)
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()
args.target = ''

datasets = com.get_all_datasets()

out_prefix = 'dataset_stats'
results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Dataset,Min,Mean,Max'
print(header, file = results_out)

def read_lengths(f, n):
    data_lines = [int(line.strip()) for line in open(f, "r", errors='replace')]
    return data_lines[0:n]

def get_stats(l):
    return min(l), sum(l) / len(l), max(l)

for dataset in datasets:
    ds_file = com.get_dataset_file(dataset)
    lengths = read_lengths(ds_file, args.max_batches * args.batch_size)
    m1, m2, m3 = get_stats(lengths)
    out_str = '%s,%d,%d,%d' % (dataset, m1, m2, m3)
    print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
