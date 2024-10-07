import gc
import sys
import utils
import argparse
import os
import numpy as np
np.random.seed(0)

def stats(arr):
    # return np.min(arr), np.mean(arr), np.max(arr)
    # return np.min(arr), np.max(arr)
    return np.mean(arr)

dataset_files = {
    "wiki_128": "/old_wikipedia/full_lengths_128.txt",
    "wiki_512": "/old_wikipedia/full_lengths_512.txt",
    "squadv2": "/squadv2/train_lengths.txt",
    "mnli": "/glue_data/MNLI/train_lengths.txt",
    "mrpc": "/glue_data/MRPC/train_lengths.txt",
    "cola": "/glue_data/CoLA/train_lengths.txt",
    "xnli": "/glue_data/XNLI/train_lengths.txt",
    "race": "/race/train_lengths.txt",
}

dataset_max_lens = {
    "race" : 512,
    "wiki_512" : 512,
    "squadv2" : 384,
    "wiki_128" : 128,
    "mnli" : 128,
    "xnli" : 128,
    "mrpc" : 112,
    "cola" : 48,
}

DATASETS = list(dataset_max_lens.keys())

MODULE_DIR = os.path.dirname(os.path.realpath(__file__)) + '/bert_layer/tvm/genlibs/'
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/data/'

def get_arm_target():
    return "llvm -mcpu=cortex-a76 -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+fullfp16,+fp-armv8,+dotprod,+crc,+crypto,+neon"

def get_cmd_parser(no_options=False):
    parser = argparse.ArgumentParser()
    if not no_options:
        parser.add_argument('--target', nargs='?', default='llvm')
        parser.add_argument('--dtype', dest='dtype', nargs='?', default='float32')
        parser.add_argument('--max-batches', dest='max_batches', default=1, type=int)
        parser.add_argument('--batch-sizes', dest='batch_sizes', nargs='+', default=[32], type=int)
        parser.add_argument('--debug', dest='debug', default=False, action='store_true')
        parser.add_argument('--disable-assert', dest='disable_assert', default=False, action='store_true')
        parser.add_argument('--debug-code', dest='debug_code', default=None, type=str)
        parser.add_argument('--debug-functions', dest='debug_functions', default=False, action='store_true')
        parser.add_argument('--manual-code', dest='manual_code', default=False, action='store_true')
        parser.add_argument('--dense-storage', dest='dense_storage', default=False, action='store_true')
        parser.add_argument('--gen-lib', dest='gen_lib', default=False, action='store_true')
        parser.add_argument('--skip-residual', dest='skip_residual', default=False, action='store_true')
        parser.add_argument('--layout-unfused', dest='layout_unfused', default=False, action='store_true')
        parser.add_argument('--dataset', nargs='?', default='random')
        parser.add_argument('--only-prep-code', dest='only_prep_code', default=False, action='store_true')
        parser.add_argument('--no-raggedness', dest='no_raggedness', default=False, action='store_true')
    return parser

def prefix_sum(extent, fn):
    s = 0
    for i in range(extent):
        s += fn(i)
    return s

def get_dataset_max_len(dataset):
    if dataset.startswith("random"):
        _, _, max_seq_len = dataset.split("_")
        return int(max_seq_len)
    else:
        return dataset_max_lens[dataset]

def get_maxlen_padded(dataset):
    return max(64, utils.ceilmult(get_dataset_max_len(dataset), 64))

def random_lengths(batch_size, avg_seq_len, max_seq_len):
    min_seq_len = 2 * avg_seq_len - max_seq_len
    return np.random.randint(min_seq_len, max_seq_len + 1, batch_size, "int32")

def int_shape(expr_shape, rmap):
    import tvm
    from tvm.tir import ir_pass
    shape = []
    for e in expr_shape:
        e1 = e
        # print(1, e1, rmap)
        e2 = ir_pass.Substitute(e1, rmap)
        # print(2, e2)
        e3 = ir_pass.Simplify(e2)
        # print(3, e3)
        shape.append(int(e3))
    return shape

def get_shape(t, rmap):
    import tvm
    if isinstance(t, tuple) or isinstance(t, list):
        return t
    elif isinstance(t, tvm.te.Tensor):
        return int_shape(t.shape, rmap)
    elif isinstance(t, tvm.tir.Buffer):
        return int_shape(t.shape.dense_shape(), rmap)
    else:
        print(t)
        assert False

def create_ragged_array(dense_shape, flat_size, dtype, ctx):
    # print("YO1: ", flat_size)
    import tvm
    # src_np_array = np.random.default_rng().random((flat_size,), dtype=np.float32)
    src_np_array = np.random.normal(size=(flat_size,)).astype(dtype)
    # src_np_array = np.full((flat_size,), 0.1, dtype).astype(dtype)
    tvm_array = tvm.nd.ragged_empty(dense_shape, flat_size, dtype=dtype, ctx=ctx)
    tvm_array.copyfrom(src_np_array, is_dst_ragged=True)
    del src_np_array
    return tvm_array

def create_numpy_array(t, dtype, rmap={}, lw_args=None):
    shape = get_shape(t, rmap)
    # print("YO2: ", shape)
    # return np.zeros(shape, dtype)
    # return np.full(shape, 0.1, dtype)
    return np.random.normal(size=shape, loc=0, scale=4).astype(dtype)

def create_tvm_array(t, dtype, ctx, rmap={}, lw_args=None):
    import tvm
    shape = get_shape(t, rmap)

    assert (lw_args is not None)
    if t in lw_args:
        flat_size = lw_args[t]
        # print(t, flat_size, shape)
        return create_ragged_array(shape, flat_size, dtype, ctx)

    # return np.zeros(shape, dtype)
    # return tvm.nd.array(np.full(shape, 0.1, dtype), ctx)
    # print("YO3: ", shape)
    # np_array = np.random.default_rng().random(shape, dtype=np.float32)
    np_array = np.random.normal(size=shape).astype(dtype)
    # np_array = np.full(shape, 0.1, dtype)
    tvm_array = tvm.nd.array(np_array, ctx)
    del np_array
    return tvm_array
    # return tvm.nd.array(np.random.sample(size=shape), ctx)

def get_ctx(target):
    import tvm
    ctx = None
    if target.startswith('llvm') or target == 'c':
        ctx = tvm.cpu(0)
    elif target.startswith('cuda'):
        ctx = tvm.gpu(0)
    else:
        raise ValueError('Unsupported target %s' % target)
    return ctx

def mean(l):
    return sum(l) / len(l)

def execute(target, built, inputs, ctx, debug = False):
    if debug:
        if target == 'c':
            built['default_function'](*inputs)
        else:
            built(*inputs)
        ctx.sync()
        return -100000000
    else:
        if target == 'c':
            built['default_function'](*inputs)
            return -100000000
            evaluator = built.time_evaluator('default_function', ctx, 1, repeat=10)
        else:
            # evaluator = built.time_evaluator(built.entry_name, ctx, repeat=5, number=20)
            evaluator = built.time_evaluator(built.entry_name, ctx, repeat=5, number=100)
        eval_result = evaluator(*inputs)
        return mean(list(eval_result.results)[1:]) * 1000
        # return mean(list(eval_result.results)) * 1000

def chunks(lst, n, m):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, min(m * n, len(lst)), n):
        yield np.array(lst[i:i + n], "int32")

def read_lengths(filename, skip = 0):
    data_lines = [int(line.strip()) for line in open(filename, "r", errors='replace')]
    return data_lines[skip:]

def read_and_chunk_lengths(batch_size, max_batches, lengths_file):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_lines = read_lengths(lengths_file)
    return list(chunks(data_lines, batch_size, max_batches))

def read_and_chunk_gemm_dims(batch_size, max_batches, filename):
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_lines = [line.strip().split(' ') for line in open(filename, "r", errors='replace')]
    ms = [int(l[0]) for l in data_lines]
    ns = [int(l[1]) for l in data_lines]
    ks = [int(l[2]) for l in data_lines]
    return (list(chunks(ms, batch_size, max_batches)),
            list(chunks(ns, batch_size, max_batches)),
            list(chunks(ks, batch_size, max_batches)))

def is_ragged(t):
    import tvm
    if isinstance(t, tvm.te.Tensor):
        if t.op.output_layout(0) is None:
            return False
        else:
            return t.op.output_layout(0).is_ragged()
    elif isinstance(t, tvm.tir.Buffer):
        return t.shape.is_ragged()
    else:
        print(t)
        assert False

def get_nlp_batches(batch_size, num_batches, dataset):
    if dataset.startswith("random"):
        _, avg_seq_len, max_seq_len = dataset.split("_")
        return [random_lengths(batch_size, int(avg_seq_len), int(max_seq_len)) for i in range(num_batches)]
    else:
        return read_and_chunk_lengths(batch_size, num_batches, DATA_DIR + "/" + dataset_files[dataset])

def run(built, i_inputs_tensors, t_inputs_tensors, batch_size, num_batches, dataset, datadir, target, debug):
    import tvm
    ctx = get_ctx(target)
    cpu_ctx = get_ctx("llvm")
    host_i_inputs, dev_i_inputs = [], []
    if len(i_inputs_tensors) == 2:
        host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), cpu_ctx) for i in i_inputs_tensors[0]]
        dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors[1]]
    t_inputs = [tvm.nd.array(create_numpy_array(i, "float32"), ctx) for i in t_inputs_tensors]
    if debug: num_batches = 1

    batches = get_nlp_batches(args.batch_size, num_batches, args.dataset)

    time = 0
    for batch in batches:
        sorted(batch)
        l_inputs = [tvm.nd.array(batch, cpu_ctx)]
        inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
        time += execute(target, built, inputs, ctx, debug)

    print("RESULTS", time / len(batches), sep=',')
    return [t.asnumpy() for t in t_inputs], batches

def reverse_sort_batches(batches):
    ret = []
    for batch in batches:
        ret.append(np.sort(batch)[::-1].astype('int32'))
    return ret

def append_padded_sum(batches, factor):
    ret = []
    for batch in batches:
        batch_sum = np.sum(batch)
        padding_length = utils.ceilmult(batch_sum, factor) - batch_sum
        if padding_length == 0: padding_length = factor
        padded = np.append(batch, padding_length).astype('int32')
        # print('PADDING', padding_length, batch_sum, np.sum(padded))
        ret.append(padded)
    return ret

def run2(built, i_inputs_tensors, t_inputs_tensors, lw_args, args, pad_sum=None):
    import tvm
    ctx = get_ctx(args.target)
    cpu_ctx = get_ctx("llvm")
    host_i_inputs, dev_i_inputs = [], []
    if len(i_inputs_tensors) == 2:
        host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), cpu_ctx) for i in i_inputs_tensors[0]]
        dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors[1]]

    num_batches = args.max_batches
    if args.debug: num_batches = 1

    batches = get_nlp_batches(args.batch_size, num_batches, args.dataset)
    if pad_sum: batches = add_padded_sum(batches, pad_sum)

    time = 0
    for batch in batches:
        sorted(batch)
        t_inputs = [create_tvm_array(i, "float32", ctx, lw_args=lw_args([batch])) for i in t_inputs_tensors]
        l_inputs = [tvm.nd.array(batch, cpu_ctx)]
        inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
        time += execute(args.target, built, inputs, ctx, args.debug)
    print("RESULTS", time / len(batches), sep=',')


    for i in range(len(t_inputs)):
        size_fn = lw_args([batch])
        target = None
        if t_inputs_tensors[i] in size_fn:
            target = np.empty(size_fn[t_inputs_tensors[i]], dtype='float32')
        t_inputs[i] = t_inputs[i].asnumpy(target=target, is_src_ragged=is_ragged(t_inputs_tensors[i]))
    return t_inputs, batches


def get_bert_layer_run_fn(bs_var):
    import tvm
    print('BS_VAR', bs_var)
    def bert_layer_run(built, i_inputs_tensors, t_inputs_tensors, lw_args, args, pad_sum=None):
        ctx = get_ctx(args.target)
        cpu_ctx = get_ctx("llvm")

        for batch_size in args.batch_sizes:
            rmap = { bs_var: batch_size }
            host_i_inputs, dev_i_inputs = [], []
            if len(i_inputs_tensors) == 2:
                host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32", rmap=rmap), cpu_ctx) for i in i_inputs_tensors[0]]
                dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32", rmap=rmap), ctx) for i in i_inputs_tensors[1]]

            num_batches = args.max_batches
            if args.debug: num_batches = 1

            batches = get_nlp_batches(batch_size, num_batches, args.dataset)
            batches = [sorted(batch, reverse=True) for batch in batches]
            if pad_sum: batches = append_padded_sum(batches, pad_sum)

            time = 0
            for batch in batches:
                t_inputs = ([batch_size] +
                            [create_tvm_array(i, "float32", ctx, rmap=rmap, lw_args=lw_args([batch]))
                             for i in t_inputs_tensors[1:]])
                if args.no_raggedness or (hasattr(args, 'full_dense') and args.full_dense):
                    l_inputs = [tvm.nd.array(batch, ctx)]
                else:
                    l_inputs = [tvm.nd.array(batch, cpu_ctx)]
                inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
                time += execute(args.target, built, inputs, ctx, args.debug)
            gc.collect()
            print("RESULTS", batch_size, time / len(batches), sep=',')
            # print(host_i_inputs[0].asnumpy())
            # print(dev_i_inputs[0].asnumpy())
            if args.debug:
                for i in range(len(t_inputs[1:])):
                    size_fn = lw_args([batch])
                    target = None
                    if t_inputs_tensors[i + 1] in size_fn:
                        target = np.empty(size_fn[t_inputs_tensors[i + 1]], dtype='float32')
                    t_inputs[i + 1] = t_inputs[i + 1].asnumpy(target=target, is_src_ragged=is_ragged(t_inputs_tensors[i + 1]))
        return t_inputs, batches
    return bert_layer_run

def get_vbatch_gemm_run_fn(bs_var, skip_m_k = False, no_scale=False):
    import tvm
    print('BS_VAR', bs_var)
    def run_vbatch_gemm(built, i_inputs_tensors, t_inputs_tensors, lw_args, args, pad_sum=None):
        ctx = get_ctx(args.target)
        cpu_ctx = get_ctx("llvm")

        for batch_size in args.batch_sizes:
            rmap = { bs_var: batch_size }
            host_i_inputs, dev_i_inputs = [], []
            if len(i_inputs_tensors) == 2:
                host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32", rmap=rmap), cpu_ctx) for i in i_inputs_tensors[0]]
                dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32", rmap=rmap), ctx) for i in i_inputs_tensors[1]]

            num_batches = args.max_batches
            if args.debug: num_batches = 1

            ms, ks, ns = read_and_chunk_gemm_dims(batch_size, num_batches, args.data_file)

            # print('Yo1')
            # sys.stdout.flush()
            if args.target == 'cuda':
                shape = get_shape(t_inputs_tensors[1], rmap)
                # np_array = np.random.normal(size=shape, loc=0, scale=4)

            # print('Yo2')
            # sys.stdout.flush()

            t_inputs = [batch_size] + [create_tvm_array(i, "float32", ctx, rmap=rmap, lw_args={}) for i in t_inputs_tensors[1:]]
            # t_inputs = [batch_size] + [tvm.nd.array(np_array, ctx) for i in t_inputs_tensors[1:]]
            # t_inputs = [batch_size] + [tvm.nd.empty(shape, 'float32', ctx) for i in t_inputs_tensors[1:]]
            time = 0
            for i in range(len(ms)):
                # print('Yo')
                # sys.stdout.flush()
                gc.collect()
                if not no_scale:
                    mb = np.ceil(ms[i] / args.tile_size).astype('int32')
                    nb = np.ceil(ns[i] / args.tile_size).astype('int32')
                    kb = np.ceil(ks[i] / args.tile_size).astype('int32')
                else:
                    mb = np.ceil(ms[i]).astype('int32')
                    nb = np.ceil(ns[i]).astype('int32')
                    kb = np.ceil(ks[i]).astype('int32')

                if skip_m_k:
                    l_inputs = [tvm.nd.array(mb, cpu_ctx)]
                else:
                    l_inputs = [tvm.nd.array(mb, cpu_ctx), tvm.nd.array(nb, cpu_ctx), tvm.nd.array(kb, cpu_ctx)]
                inputs = t_inputs + l_inputs + host_i_inputs + dev_i_inputs
                this_time = execute(args.target, built, inputs, ctx, args.debug)
                time += this_time
                # print(' ', this_time)
                # sys.stdout.flush()
                gc.collect()


            print("RESULTS", batch_size, time / len(ms), sep=',')
            for i in range(len(t_inputs) - 1):
                size_fn = {}
                target = None
                if t_inputs_tensors[i + 1] in size_fn:
                    target = np.empty(size_fn[t_inputs_tensors[i + 1]], dtype='float32')
                t_inputs[i + 1] = t_inputs[i + 1].asnumpy(target=target, is_src_ragged=is_ragged(t_inputs_tensors[i + 1]))
        return t_inputs, ms, ns, ks
    return run_vbatch_gemm

def run_trmm(built, i_inputs_tensors, t_inputs_tensors, lw_args, args, pad_sum=None):
    import tvm
    ctx = get_ctx(args.target)
    cpu_ctx = get_ctx("llvm")
    host_i_inputs, dev_i_inputs = [], []
    if len(i_inputs_tensors) == 2:
        host_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), cpu_ctx) for i in i_inputs_tensors[0]]
        dev_i_inputs = [tvm.nd.array(create_numpy_array(i, "int32"), ctx) for i in i_inputs_tensors[1]]

    t_inputs = [create_tvm_array(i, "float32", ctx, lw_args={}) for i in t_inputs_tensors]
    inputs = t_inputs + host_i_inputs + dev_i_inputs
    time = execute(args.target, built, inputs, ctx, args.debug)

    print("RESULTS", time, sep=',')
    for i in range(len(t_inputs)):
        size_fn = {}
        target = None
        if t_inputs_tensors[i] in size_fn:
            target = np.empty(size_fn[t_inputs_tensors[i]], dtype='float32')
        t_inputs[i] = t_inputs[i].asnumpy(target=target, is_src_ragged=is_ragged(t_inputs_tensors[i]))
    return t_inputs

def lower_or_build(name, s, inputs, args, prep_code_mode='with_prep_code', binds=None,
                   size_fn={}, pad_sum=None, substitutes=None, run_function=run2, hoist_loads=False):
    import tvm
    prep_code_mode = 'only_prep_code' if args.only_prep_code else prep_code_mode
    with tvm.build_config(prep_code_mode=prep_code_mode,
                          fill_in_function_bodies=not args.debug_functions,
                          hoist_loads=hoist_loads,
                          disable_assert=args.disable_assert if hasattr(args, 'disable_assert') else False):
        if args.gen_lib:
            fadd, i_bufs = tvm.build(s, inputs, args.target, binds=binds)
            variant = ''
            if hasattr(args, 'sched'): variant = str(args.sched)
            if hasattr(args, 'padding_mode'): variant = '_' + str(args.padding_mode)
            fadd.export_library(MODULE_DIR + name + variant + '.so')
            with open(MODULE_DIR + name + variant + '_bufs.txt', 'w') as buf_file:
                for buf in i_bufs[0]:
                    print('h', buf.shape.dense_shape(), buf.dtype, file=buf_file, sep='|')
                for buf in i_bufs[1]:
                    print('d', buf.shape.dense_shape(), buf.dtype, file=buf_file, sep='|')
            return None, None
        else:
            if args.debug_code == 'ir':
                lowered = tvm.lower(s, inputs, args.target, simple_mode=True, binds=binds, substitutes=substitutes)
                print(lowered)
                return None, None
            elif args.debug_code == 'code':
                fadd, _ = tvm.build(s, inputs, args.target, binds=binds)
                if args.target == 'cuda':
                    print('-----GPU code-----\n' + fadd.imported_modules[0].get_source())
                else:
                    print('-----CPU code-----\n' + fadd.get_source())
                return None, None
            else:
                assert args.debug_code is None
                fadd, i_bufs = tvm.build(s, inputs, args.target, binds=binds, substitutes=substitutes)
                # fadd = tvm.runtime.module.load_module('/home/ppf/rnn_compilers/ragged_tensors/incubator-tvm/build/attn_v.so')
                return run_function(fadd, i_bufs, inputs[1], size_fn, args, pad_sum=pad_sum)
