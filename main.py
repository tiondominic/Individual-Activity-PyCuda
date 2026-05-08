import numpy as np
import time
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule


def compare(N):
    print(f"\n===== Testing N = {N:,} =====")

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)

    start = time.perf_counter()

    c_cpu = a + b

    end = time.perf_counter()

    cpu_time = end - start

    print(f"CPU Time: {cpu_time:.6f} seconds")

    mod = SourceModule("""
    __global__ void add_vectors(float *dest, float *a, float *b, int N)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (idx < N)
        {
            dest[idx] = a[idx] + b[idx];
        }
    }
    """, options=[
        "--allow-unsupported-compiler",
        "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH"
    ])
    #Setup like this to avoid errors

    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(a.nbytes)

    # Transfer data CPU -> GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    func = mod.get_function("add_vectors")

    block_size = 256
    grid_size = (N + block_size - 1) // block_size

    start = time.perf_counter()

    func(
        c_gpu,
        a_gpu,
        b_gpu,
        np.int32(N),
        block=(block_size, 1, 1),
        grid=(grid_size, 1)
    )

    cuda.Context.synchronize()

    end = time.perf_counter()

    gpu_time = end - start

    print(f"GPU Time: {gpu_time:.6f} seconds")

    c_result = np.empty_like(a)

    cuda.memcpy_dtoh(c_result, c_gpu)

    match = np.allclose(c_cpu, c_result)

    print("Results Match:", match)

    if gpu_time < cpu_time:
        speedup = cpu_time / gpu_time
        print(f"GPU is {speedup:.2f}x faster")
    else:
        slowdown = gpu_time / cpu_time
        print(f"CPU is {slowdown:.2f}x faster")

    a_gpu.free()
    b_gpu.free()
    c_gpu.free()


sizes = [
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000
]

for N in sizes:
    compare(N)