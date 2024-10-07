import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
CUBLAS_RUNNER = SCRIPT_DIR + '/../trmm/cublas/gemm_cublas'
MKL_RUNNER = SCRIPT_DIR + '/../trmm/mkl/trmm'
TVM_GPU_RUNNER = SCRIPT_DIR + '/../trmm/tvm/trmm.py'
TVM_CPU_RUNNER = SCRIPT_DIR + '/../trmm/tvm/trmm_cpu.py'
TACO_RUNNER = SCRIPT_DIR + '/../taco/taco_csr_trmm'
PYTHON = 'python3'

def get_mkl_runner(pad):
    def run_mkl(m_size, n_size, err_file, args):
        assert m_size == n_size
        cmd = [MKL_RUNNER, str(m_size), '1' if pad else'0', str(100), str(1)]
        out, err = run_cmd(cmd)
        if err: print(err, file = err_file)
        return com.extract_times(out, 1)[0]
    return run_mkl

def get_cublas_runner(pad):
    def run_cublas(m_size, n_size, err_file, args):
        cmd = [CUBLAS_RUNNER, str(m_size), str(n_size), '1' if pad else'0', str(100), str(1)]
        out, err = run_cmd(cmd)
        if err: print(err, file = err_file)
        return com.extract_times(out, 1)[0]
    return run_cublas

def get_tvm_runner(balance, split):
    def run_tvm(m_size, n_size, err_file, args):
        runner = TVM_GPU_RUNNER if args.target == "cuda" else TVM_CPU_RUNNER
        cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--m', str(m_size), '--n', str(n_size)]
        if balance: cmd += ['--load-balance']
        if split: cmd += ['--op-split']
        print(' '.join(cmd))
        out, err = run_cmd(cmd)
        if err: print(err, file = err_file)
        return com.extract_times(out, 1)[0]
    return run_tvm

def taco_runner(m_size, n_size, err_file, args):
    runner = TACO_RUNNER
    cmd = [runner, str(m_size), '0']
    out, err = run_cmd(cmd)
    if err: print(err, file = err_file)
    return com.extract_times(out, 1)[0]

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

# ops = ['Sq', 'Th']
ops = ['Sq']
op_m_sizes = {
    'Sq': [512, 1024, 2048, 4096, 8192],
    # 'Th': [128, 256, 512, 1024, 2048, 4096, 8192, 8192*2, 8192*4],
}

def get_op_n_size(op, m):
    if op == 'Sq': return m_size
    else: return 128

targets = [args.target] if args.target else ['cuda']

if args.target == 'cuda':
    framework_funs = {
        # 'cublas_nopad': get_cublas_runner(False),
        # 'cublas_pad': get_cublas_runner(True),
        # 'cora_unsplit': get_tvm_runner(False, False),
        # 'cora_unbalanced': get_tvm_runner(False, True),
        # 'cora_balanced': get_tvm_runner(True, True),
        'taco': taco_runner,
    }
else:
    framework_funs = {
        'mkl_nopad': get_mkl_runner(False),
        'mkl_pad': get_mkl_runner(True),
        'cora_unsplit': get_tvm_runner(False, False),
        # 'cora_unbalanced': get_tvm_runner(False, True),
        # 'cora_balanced': get_tvm_runner(True, True),
    }


results_out, results_err = get_out_files(args, 'trmm', 'a' if args.append else 'w')
header = 'Op,Target,M,N'
for framework, func in framework_funs.items(): header += ',' + framework + ' (ms)'
print(header, file = results_out)

for op in ops:
    for target in targets:
        exe_times = {}
        for m_size in op_m_sizes[op]:
            n_size = get_op_n_size(op, m_size)
            for framework, func in framework_funs.items():
                log(args, 'Running %s %s %s %d %d' % (op, target, framework, m_size, n_size))
                exe_times[framework] = func(m_size, n_size, results_err, args)
                print(exe_times[framework])

            out_str = '%s,%s,%d,%d' % (op, target, m_size, n_size)
            for framework, framework_exe_time in exe_times.items():
                out_str += ',%g' % framework_exe_time
            print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
