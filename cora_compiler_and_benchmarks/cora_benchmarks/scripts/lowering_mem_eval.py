import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MEM_RUNNER = SCRIPT_DIR + '/../lowering_mem_eval/mem'

def execute(b_size, op, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    cmd = [MEM_RUNNER, str(b_size), str(n_batch), com.get_dataset_file(dataset), op]
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    print(out)
    ftime, ttime = com.extract_times(out, 2)
    fmem, tmem = com.extract_mem(out, 2)
    return (fmem, ftime, tmem, ttime)

parser = argparse.ArgumentParser()
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

# batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
# batch_sizes = [8]
b_sizes = [32, 64, 128]
args.target = 'cuda'
datasets = com.get_all_datasets() if args.dataset is None else [args.dataset]

ops = ['csf', 'cora']
out_prefix = 'lowering_mem'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Dataset,Batch Size,csf_ftime,csf_fmem,csf_ttime,csf_tmem,cora_ftime,cora_fmem,cora_ttime,cora_tmem'
print(header, file = results_out)

for dataset in datasets:
    for b_size in b_sizes:
        out_str = '%s,%d' % (dataset, b_size)
        for op in ops:
            fmem, ftime, tmem, ttime = execute(b_size, op, dataset, args.max_batches, results_err, args)
            out_str += ',%g,%g,%g,%g' % (ftime, fmem, ttime, tmem)
        print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
