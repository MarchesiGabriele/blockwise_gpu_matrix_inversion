import numpy as np
from time import perf_counter as pt
import opencl_matmul as mm
import sys
import pyopencl as cl
import os

#os.environ['OMP_NUM_THREADS'] = '1'

N = 2048 
np.random.seed(np.int64(pt()))

# TEST OPENCL MATMUL 
def test_mat_mul():
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # numpy
    start = pt() 
    #m0 = np.dot(A,B) 
    m0 = A@B 
    end = pt() 
    print(f"Tempo Numpy: {end-start}s")
    print(f"GFLOPS Numpy: {(N**3 * 2)/((end-start)*10**9)} GFLOPS")

    #opencl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    start = pt() 
    m1 = mm.matmul(A, B, N, FP32, ctx, queue)
    end = pt() 
    #print(m0)
    print()

    #print(m1)
    print(f"Tempo OpenCL: {end-start}s")
    print(f"OpenCL GFLOPS: {(N1*N2*2*N)/((end-start)*1e9)} GFLOPS")
    print("Errore medio Numpy vs OpenCL: ", np.sum(np.subtract(m1, m0))/(N*N))

    sys.exit(0)


if __name__ == "__main__":

    platforms = cl.get_platforms()

    for i, platform in enumerate(platforms):
        print(f"Platform {i}: {platform.name}")

        # Ottieni tutti i dispositivi di questa piattaforma
        devices = platform.get_devices()
        for j, device in enumerate(devices):
            print(f"  Device {j}: {device.name}")
            print(f"    Type: {'GPU' if device.type == cl.device_type.GPU else 'CPU'}")
            print(f"    Compute Units: {device.max_compute_units}")
            print(f"    Global Memory: {device.global_mem_size / 1e6} MB")
            print(f"    Vendor: {device.vendor}\n")

    
    mygpu = platforms[0].get_devices()[0]















