import math
import os

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

def get_tvm_callback_cuda_postproc(args, path, dirname = 'perf', fileprefix = 'dummy_file'):
    import tvm
    from tvm.contrib import nvcc, cublas, cblas
    def tvm_callback_cuda_postproc(code):
        d = os.path.dirname(path)
        d = d + '/' + dirname + '/'
        if not os.path.exists(d):
            os.mkdir(d)
        write_code(code, d + fileprefix + "_gen.cu")
        if args.manual_code:
            # print("Using manual code")
            code = open(d + fileprefix + "_manual.cu").read()
        return code
    return tvm_callback_cuda_postproc

def get_tvm_callback_cuda_compile(threads, grid_sync = False):
    import tvm
    from tvm.contrib import nvcc, cublas, cblas
    tvm.target.set_cuda_grid_sync_on(grid_sync)
    tvm.runtime.module.set_cuda_grid_sync_on(grid_sync)
    def tvm_callback_cuda_compile(code):
        print('Using NVCC')
        # options = ["--ptxas-options='-v -warn-lmem-usage -warn-spills' --nvlink-options='-v'", '-rdc=true']
        # options = ["-Xcompiler", "-rdynamic", "-D_FORCE_INLINES", "--use_fast_math", "-lineinfo",
                   # "--ptxas-options='-allow-expensive-optimizations'", "--maxrregcount=" + str((65536 // threads) - 1)]
        # options = ["-Xcompiler", "-rdynamic", "-D_FORCE_INLINES",
                   # "--ptxas-options='-allow-expensive-optimizations'", "--maxrregcount=" + str((65536 // threads) - 1)]
        # options = ["-lineinfo", "-Xcompiler", "-rdynamic", "-D_FORCE_INLINES",
                   # "--ptxas-options='-allow-expensive-optimizations'", "--use_fast_math"]
        # options = ["-Xcompiler", "-rdynamic", "-D_FORCE_INLINES", "--use_fast_math"]
        options = ["-Xcompiler", "-rdynamic", "-D_FORCE_INLINES"]
        # options = ["-lineinfo", "-Xcompiler", "-rdynamic", "-D_FORCE_INLINES",
                   # "--ptxas-options='-allow-expensive-optimizations'", "--use_fast_math"]
        if nvcc.have_grid_sync(grid_sync): options += ["-rdc=true", "-L /usr/lib/x86_64-linux-gnu"]
        ptx = nvcc.compile_cuda(code, target="cubin", options = options)
        return ptx
    return tvm_callback_cuda_compile

def ceildiv(a, b):
    import tvm
    if isinstance(a, int) and isinstance(b, int):
        return (a + b - 1) // b
    else:
        return tvm.floordiv(a + b - 1, b)

def ceilmult(a, b):
    import tvm
    try:
        ai = int(a)
        bi = int(b)
        return bi * ((ai + bi - 1) // bi)
    except:
        return b * tvm.floordiv(a + b - 1, b)

def floormult(a, b):
    import tvm
    if isinstance(a, int) and isinstance(b, int):
        return b * (a // b)
    else:
        return b * tvm.floordiv(a, b)

def gelu(x):
    import tvm
    cdf = 0.5 * (1.0 + tvm.tanh((0.7978845608028654 * (x + 0.044715 * x * x * x))))
    return x * cdf;

def next_power_of_2(x):
    return 1 if x == 0 else 2**math.ceil(math.log2(x))
