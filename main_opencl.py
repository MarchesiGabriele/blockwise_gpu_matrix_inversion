import blocking_algo_testing as bl
import numpy as np
import time
import opencl_matmul as mm
import sys

# TODO: Sostituire le moltiplicazioni tra matrici con moltiplicazioni eseguite sulla GPU

N = 4096 

FP32 = True 
np.random.seed(0)

# TEST OPENCL MATMUL 
def test_mat_mul():
    if FP32:
        A = np.random.rand(N//3, N).astype(np.float32)
        B = np.random.rand(N, N//2).astype(np.float32)
        A = np.matrix.round(A, 3)
        B = np.matrix.round(B, 3)
    else:
        A = np.random.rand(N//3, N)
        B = np.random.rand(N, N//2)

    #numpy
    start = time.monotonic()
    m0 = np.dot(A,B) 
    end = time.monotonic()
    print(f"Tempo NUMPY MATMUL: {end-start}s")
    
    #opencl
    start = time.monotonic()
    m1 = mm.matmul(A, B, N//3, N, N//2, FP32)
    end = time.monotonic()
    print("NUMP")
    print(m0)
    print()

    print("OPEN")
    print(m1)
    print(f"Tempo OPENCL MATMUL: {end-start}s")
    print("errore medio numpy vs opencl: ", np.sum(np.subtract(m1, m0))/(N*N))

    sys.exit(0)


if __name__ == "__main__":

    # Matrice iniziale
    P = np.random.rand(N, N)
    print("primo elemento: ", P[0][0], "\n")

    test_mat_mul()
    
    start = time.monotonic()
    inversa = bl.inversa(P)
    end = time.monotonic()

    print(f"Tempo: {end-start}")

    res = inversa@P
    frobenius_norm = np.sqrt(np.sum(res*res))
    print(f"Errore: {np.sqrt(N)-frobenius_norm}")















