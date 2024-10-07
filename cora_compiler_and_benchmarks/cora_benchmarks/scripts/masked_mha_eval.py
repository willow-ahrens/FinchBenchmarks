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

def generate_tvm_libs(framework, dataset, args):
    if 'cora' in framework:
        cmd = [TVM_LIB_RUNNER, dataset,
               '0',
               '1' if ('masked' in framework) else '0',
               '0']
        print(' '.join(cmd))
        out, err = run_cmd(cmd)
        print(out, err)

def run_pytorch(b_size, dataset, n_batch, err_file, args):
    log(args, ' Batch size %d' % (b_size))
    cmd = [PYTHON, PYTORCH_RUNNER, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
           '--max-batches', str(n_batch), '--dataset', dataset]
    cmd += ['--masked-mha']
    print(' '.join(cmd))
    out, err = '', ''
    out, err = run_cmd(cmd)
    print(out)
    if err: print(err, file = err_file)

    return com.extract_times(out, 1)[0]

def get_tvm_runner(masked):
    def run_tvm(b_size, dataset, n_batch, err_file, args):
        log(args, ' Batch size %d' % (b_size))
        runner = TVM_EXE_RUNNER

        cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
               '--max-batches', str(n_batch), '--dataset', dataset]

        if masked: cmd += ['--masked-mha']
        else: cmd += ['--plain-mha']
        print(' '.join(cmd))
        out, err = '', ''
        out, err = run_cmd(cmd)
        print(out)
        if err: print(err, file = err_file)

        return com.extract_times(out, 1)[0]
    return run_tvm;

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default='cuda')
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--gen-libs', dest='gen_libs', default=False, action='store_true')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

batch_sizes = [32, 64, 128]
target = 'cuda'
# datasets = com.cluster_datasets_by_max_len() if args.dataset is None else {com.get_dataset_max_len(args.dataset) : [args.dataset]}
# datasets = {512:['race', 'wiki_512'],384:['squadv2'],128:['wiki_128','mnli','xnli'],112:['mrpc'],48:['cola']}
datasets = {512:['race'],128:['mnli']}

framework_funs = {
    'pytorch': lambda b_sizes, *args: com.batchify(b_sizes, run_pytorch, *args),
    'cora_plain': lambda b_sizes, *args: com.batchify(b_sizes, get_tvm_runner(False), *args),
    'cora_masked': lambda b_sizes, *args: com.batchify(b_sizes, get_tvm_runner(True), *args),
}

out_prefix = 'bert_layer_mmha'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Target,Dataset,Batch Size'
for framework, func in framework_funs.items(): header += ',' + framework + ' (ms)'
print(header, file = results_out)

exe_times = {}
for framework, func in framework_funs.items():
    ds_exe_times = {}
    for _, dataset_list in datasets.items():
        if args.gen_libs: generate_tvm_libs(framework, dataset_list[0], args);
        for dataset in dataset_list:
            # log(args, 'Running %s %s %s %s' % (target, dataset, framework, batch_sizes))
            print('Running %s %s %s %s' % (target, dataset, framework, batch_sizes))
            ds_exe_times[dataset] = func(batch_sizes, dataset, args.max_batches, results_err, args)
            # print(ds_exe_times[framework])
    exe_times[framework] = ds_exe_times
print(exe_times)

for _, dataset_list in datasets.items():
    for dataset in dataset_list:
        for b_size in batch_sizes:
            out_str = '%s,%s,%d' % (target, dataset, b_size)
            for framework in framework_funs:
                out_str += ',%g' % exe_times[framework][dataset][b_size]
            print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
