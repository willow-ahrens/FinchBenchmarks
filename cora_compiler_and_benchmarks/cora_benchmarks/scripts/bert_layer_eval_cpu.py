import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PYTORCH_RUNNER_CPU = SCRIPT_DIR + '/../bert_layer/pytorch/layer_cpu.py'
PYTORCH_RUNNER_GPU = SCRIPT_DIR + '/../bert_layer/pytorch/layer.py'
TVM_GPU_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/masked_mha.py'
TVM_CPU_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/masked_mha_cpu.py'
TVM_MEM_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/bert_layer_memory_inplace.py'
TVM_GPU_LIB_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/gen_libs.sh'
TVM_CPU_LIB_RUNNER = SCRIPT_DIR + '/../bert_layer/tvm/gen_libs_cpu.sh'
FTRANS_EXE_RUNNER = SCRIPT_DIR + '/../bert_layer/faster_transformer/run_encoder_sample.sh'
FTRANS_MEM_RUNNER = SCRIPT_DIR + '/../bert_layer/faster_transformer/mem/memory'
PYTHON = 'python3'

def generate_tvm_libs(dataset, target, args):
    if target == "cpu": runner = TVM_CPU_LIB_RUNNER
    else: runner = TVM_GPU_LIB_RUNNER
    cmd = [runner, dataset,
           '1' if args.bin_packed else '0',
           '0',
           '1' if args.prep_overhead else '0']
    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    print(out, err)

def run_pytorch(b_size, dataset, n_batch, err_file, args):
    print(args.target, args.target == "cpu")
    if args.target == "cpu": runner = PYTORCH_RUNNER_CPU
    else: runner = PYTORCH_RUNNER_GPU

    log(args, ' Batch size %d' % (b_size))
    cmd = [PYTHON, runner, '--target', target, '--batch-size', str(b_size),
           '--max-batches', str(n_batch), '--dataset', dataset]
    if args.mem: cmd += ['--mem']
    if args.target == "cpu": cmd += ['--masked-mha']

    print(' '.join(cmd))
    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)

    if args.mem: return com.extract_mem(out)
    else: return com.extract_times(out, 1)[0]

def get_ftrans_runner(no_pad):
    def run_ftrans(b_size, dataset, n_batch, err_file, args):
        log(args, ' Batch size %d' % (b_size))
        num_layers = 25

        if args.mem:
            cmd = [FTRANS_MEM_RUNNER, str(b_size), com.get_dataset_file(dataset)]
        else:
            cmd = [FTRANS_EXE_RUNNER, com.get_dataset_file(dataset), str(b_size), str(n_batch),
                   str(com.get_dataset_max_len(dataset)), str(num_layers), '1' if no_pad else '0']

        # print(' '.join(cmd))
        out, err = run_cmd(cmd)
        # print(out)
        if err: print(err, file = err_file)

        if args.mem: return com.extract_mem(out)
        else: return com.extract_times(out, 1)[0]
    return run_ftrans

# def get_cora_runner(balance):
#     def run_cora(b_size, dataset, n_batch, err_file, args):
#         log(args, ' Batch size %d' % (b_size))
#         if args.target == "cpu": runner = TVM_CPU_EXE_RUNNER
#         else:
#             runner = TVM_MEM_RUNNER if args.mem else TVM_CPU_EXE_RUNNER

#         cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--batch-size', str(b_size),
#                '--max-batches', str(n_batch), '--dataset', dataset]
#         if args.bin_packed: cmd += ['--bin-packed']
#         if balance: cmd += ['--average']
#         if args.target == "cpu": cmd += ['--masked-mha']
#         print(' '.join(cmd))
#         out, err = '', ''
#         out, err = run_cmd(cmd)
#         print(out)
#         if err: print(err, file = err_file)

#         if args.mem: return com.extract_mem(out)
#         else: return com.extract_times(out, 1)[0]
#     return run_cora

def run_cora(b_sizes, dataset, n_batch, err_file, args):
    runners = [
        SCRIPT_DIR + '/../bert_layer/tvm/pre_linear_cpu.py',
        SCRIPT_DIR + '/../bert_layer/tvm/post_linear_cpu.py',
        SCRIPT_DIR + '/../bert_layer/tvm/qkt_cpu.py',
        SCRIPT_DIR + '/../bert_layer/tvm/attn_v_cpu.py',
        SCRIPT_DIR + '/../bert_layer/tvm/softmax_cpu.py'
    ]

    times = {}
    for b_size in b_sizes: times[b_size] = 0.0

    for runner in runners:
        cmd = ([PYTHON, runner, '--batch-sizes'] +
               [str(i) for i in b_sizes] +
               ['--max-batches', str(n_batch), '--dataset', dataset, '--skip-residual'])

        log(args, '  DS/OP %s %s' % (dataset, runner))

        print(' '.join(cmd))
        out, err = '', ''
        out, err = run_cmd(cmd)
        # print(out)
        if err: print(err, file = err_file)

        res = com.extract_time_batches(out)
        print(res)
        for a, b in res.items(): times[a] += b
    return times

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--dataset', nargs='?', default=None)
parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
parser.add_argument('--bin-packed', dest='bin_packed', default=False, action='store_true')
parser.add_argument('--prep-overhead', dest='prep_overhead', default=False, action='store_true')
parser.add_argument('--gen-libs', dest='gen_libs', default=False, action='store_true')
parser.add_argument('--mem', dest='mem', default=False, action='store_true')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

# batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
# batch_sizes = [8]
batch_sizes = [32, 64, 128]
# batch_sizes = [32, 64]
# batch_sizes = [1, 1]
targets = [args.target] if args.target else ['cuda']
datasets = com.cluster_datasets_by_max_len() if args.dataset is None else {com.get_dataset_max_len(args.dataset) : [args.dataset]}
# datasets = {512:['race', 'wiki_512'],384:['squadv2'],128:['wiki_128','mnli','xnli'],112:['mrpc'],48:['cola']}
# datasets = {384:['squadv2'],128:['wiki_128','mnli','xnli'],112:['mrpc'],48:['cola']}
# datasets = {128:['wiki_128'],48:['cola']}

framework_funs = {
    'pytorch': lambda b_sizes, *args: com.batchify(b_sizes, run_pytorch, *args),
    'cora': run_cora,
}

out_prefix = 'bert_layer'
if args.prep_overhead: out_prefix += '_prelude'
if args.mem: out_prefix += '_mem'

results_out, results_err = get_out_files(args, out_prefix, 'a' if args.append else 'w')
header = 'Target,Dataset,Batch Size'
for framework, func in framework_funs.items(): header += ',' + framework + ' (ms)'
print(header, file = results_out)

for target in targets:
    for _, dataset_list in datasets.items():
        # if args.gen_libs: generate_tvm_libs(dataset_list[0], target, args);
        for dataset in dataset_list:
            exe_times = {}
            for framework, func in framework_funs.items():
                log(args, 'Running %s %s %s %s' % (target, dataset, framework, batch_sizes))
                exe_times[framework] = func(batch_sizes, dataset, args.max_batches, results_err, args)
                print(exe_times[framework])

            for b_size in batch_sizes:
                out_str = '%s,%s,%d' % (target, dataset, b_size)
                for framework, framework_exe_times in exe_times.items():
                    out_str += ',%g' % framework_exe_times[b_size]
                print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
