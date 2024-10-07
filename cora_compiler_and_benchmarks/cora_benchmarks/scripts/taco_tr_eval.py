import os
import sys
import common as com
from common import run_cmd, INF, get_out_files, log, run_linearization
import argparse

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TVM_RUNNER = SCRIPT_DIR + '/../taco/tradd.py'
TACO_RUNNER = SCRIPT_DIR + '/../taco/taco_csr_tradd'
PYTHON = 'python3'

runners = {
    'taco_csr': {
        'trmm': SCRIPT_DIR + '/../taco/taco_csr_trmm',
        'tradd': SCRIPT_DIR + '/../taco/taco_csr_tradd',
        'trmul': SCRIPT_DIR + '/../taco/taco_csr_trmul',
    },
    'taco_bcsr': {
        'trmm': SCRIPT_DIR + '/../taco/taco_bcsr_trmm',
        'tradd': SCRIPT_DIR + '/../taco/taco_bcsr_tradd',
        'trmul': SCRIPT_DIR + '/../taco/taco_bcsr_trmul',
    },
    'cora': {
        'trmm': SCRIPT_DIR + '/../trmm/tvm/trmm.py',
        'tradd': SCRIPT_DIR + '/../taco/trop.py',
        'trmul': SCRIPT_DIR + '/../taco/trop.py',
    },
}

def run_tvm(m_size, op, err_file, args):
    runner = runners['cora'][op]

    cmd = [PYTHON, runner, '--target', com.get_tvm_target(target), '--m', str(m_size)]
    if op == 'trmul': cmd += ['--op', 'mul']
    if op == 'tradd': cmd += ['--op', 'add']
    if op == 'trmm': cmd += ['--n', str(m_size), '--load-balance', '--op-split']
    out, err = '', ''
    out, err = run_cmd(cmd)
    print(' '.join(cmd))
    if err: print(err, file = err_file)
    return com.extract_times(out, 1)[0]

def get_taco_runner(framework):
    def run_taco(m_size, op, err_file, args):
        runner = runners[framework][op]
        cmd = [runner, str(m_size)]
        if 'bcsr' in framework: cmd += ['32']
        if 'bcsr' in framework and op ==  'tradd':
            out, err = '', ''
        else:
            out, err = run_cmd(cmd)
        print(' '.join(cmd))
        if err: print(err, file = err_file)
        return com.extract_times(out, 1)[0]
    return run_taco

parser = argparse.ArgumentParser()
parser.add_argument('--target', nargs='?', default=None)
parser.add_argument('--out-dir', dest='out_dir', nargs='?', default='perf_results')
parser.add_argument('--stdout', dest='stdout', default=False, action='store_true')
parser.add_argument('--append', dest='append', default=False, action='store_true')
args = parser.parse_args()

m_sizes = [128, 512, 2048, 8192]

targets = [args.target] if args.target else ['cuda']

ops = ['trmm', 'tradd', 'trmul']
if args.target == 'cuda':
    framework_funs = {
        'taco_csr': get_taco_runner('taco_csr'),
        'taco_bcsr': get_taco_runner('taco_bcsr'),
        'cora': run_tvm,
    }

results_out, results_err = get_out_files(args, 'taco', 'a' if args.append else 'w')
header = 'Target,M,Op'
for framework, func in framework_funs.items(): header += ',' + framework + ' (ms)'
print(header, file = results_out)

for target in targets:
    exe_times = {}
    for m_size in m_sizes:
        for op in ops:
            for framework, func in framework_funs.items():
                log(args, 'Running %s %s %d' % (framework, op, m_size))
                exe_times[framework] = func(m_size, op, results_err, args)
                print(exe_times[framework])

            out_str = '%s,%d,%s' % (target, m_size, op)
            for framework, framework_exe_time in exe_times.items():
                out_str += ',%g' % framework_exe_time
            print(out_str, file = results_out)

if not args.stdout:
    results_out.close()
    results_err.close()
