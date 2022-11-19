import blocking_algo_testing as bl
import numpy as np
from time import perf_counter as pt
import opencl_matmul as mm
import sys
import pyopencl as cl
import os

#os.environ['OMP_NUM_THREADS'] = '1'

N = 2048 

N1 = N//4
N2 = N//2

FP32 = True 
np.random.seed(np.int64(pt()))

# TEST OPENCL MATMUL 
def test_mat_mul():
    if FP32:
        A = np.random.randn(N1, N).astype(np.float32)
        B = np.random.randn(N, N2).astype(np.float32)
        A = np.matrix.round(A, 3)
        B = np.matrix.round(B, 3)
        #print(A)
        #print(B)
    else:
        A = np.random.rand(N1, N)
        B = np.random.rand(N, N2)

    #numpy
    start = pt() 
    #m0 = np.dot(A,B) 
    m0 = A@B 
    end = pt() 
    print(f"Tempo Numpy: {end-start}s")
    print(f"Numpy GFLOPS: {(N1*N2*2*N)/((end-start)*1e9)} GFLOPS")

    #opencl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    start = pt() 
    m1 = mm.matmul(A, B, N1, N, N2, FP32, ctx, queue)
    end = pt() 
    #print(m0)
    print()

    #print(m1)
    print(f"Tempo OpenCL: {end-start}s")
    print(f"OpenCL GFLOPS: {(N1*N2*2*N)/((end-start)*1e9)} GFLOPS")
    print("Errore medio Numpy vs OpenCL: ", np.sum(np.subtract(m1, m0))/(N*N))

    sys.exit(0)


if __name__ == "__main__":

    # Matrice iniziale
    P = np.random.rand(N, N)

    test_mat_mul()
    
    start = pt() 
    inversa = bl.inversa(P)
    end = pt() 

    print(f"Tempo: {end-start}")

    res = inversa@P
    frobenius_norm = np.sqrt(np.sum(res*res))
    print(f"Errore: {np.sqrt(N)-frobenius_norm}")















