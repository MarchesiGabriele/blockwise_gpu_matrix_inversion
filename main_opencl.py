import blocking_algo_testing as bl
import numpy as np
import time
import opencl_matmul as mm
import sys
import pyopencl as cl
import os
# TODO: Sostituire le moltiplicazioni tra matrici con moltiplicazioni eseguite sulla GPU

#os.environ['OMP_NUM_THREADS'] = '1'

N = 5050 

N1 = N//4
N2 = N//2

FP32 = True 
np.random.seed(np.int64(time.monotonic()))

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
    start = time.monotonic()
    #m0 = np.dot(A,B) 
    m0 = A@B 
    end = time.monotonic()
    print(f"Tempo Numpy: {end-start}s")
    print(f"Numpy GFLOPS: {(N1*N2*2*N)/((end-start)*1e9)} GFLOPS")
    
    
    #opencl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    start = time.monotonic()
    m1 = mm.matmul(A, B, N1, N, N2, FP32, ctx, queue)
    end = time.monotonic()
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
    
    start = time.monotonic()
    inversa = bl.inversa(P)
    end = time.monotonic()

    print(f"Tempo: {end-start}")

    res = inversa@P
    frobenius_norm = np.sqrt(np.sum(res*res))
    print(f"Errore: {np.sqrt(N)-frobenius_norm}")















