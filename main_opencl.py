import blocking_algo_testing as bl
import numpy as np
import time

# TODO: Sostituire le moltiplicazioni tra matrici con moltiplicazioni eseguite sulla GPU

N = 5000 


if __name__ == "__main__":

    # Matrice iniziale
    P = np.random.randint(N, size=(N,N))

    start = time.monotonic()
    inversa = bl.inversa(P)
    end = time.monotonic()

    print(f"Tempo: {end-start}")

    res = inversa@P
    frobenius_norm = np.sqrt(np.sum(res*res))
    print(f"Errore: {np.sqrt(N)-frobenius_norm}")















