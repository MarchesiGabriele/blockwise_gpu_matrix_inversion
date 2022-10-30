import blocking_algo_testing as bl
import numpy as np
import time
import opencl_matmul as mm
import sys

# TODO: Sostituire le moltiplicazioni tra matrici con moltiplicazioni eseguite sulla GPU

N = 4096 

if __name__ == "__main__":

    # Matrice iniziale
    P = np.random.rand(N, N)
    print("primo elemento: ", P[0][0], "\n")
  
    
    # TEST OPENCL MATMUL 
    #####################################################
    start = time.monotonic()
    np.dot(P,P) 
    end = time.monotonic()
    print(f"Tempo NUMPY MATMUL: {end-start}s")

    start = time.monotonic()
    m1 = mm.matmul(P,P,N)
    end = time.monotonic()
    print(f"Tempo OPENCL MATMUL: {end-start}s")
    sys.exit(0)
    #####################################################

    start = time.monotonic()
    inversa = bl.inversa(P)
    end = time.monotonic()

    print(f"Tempo: {end-start}")

    res = inversa@P
    frobenius_norm = np.sqrt(np.sum(res*res))
    print(f"Errore: {np.sqrt(N)-frobenius_norm}")















