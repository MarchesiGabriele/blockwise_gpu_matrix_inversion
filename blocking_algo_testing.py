import numpy as np
from numpy.linalg import inv

# TODO: Sostituire le moltiplicazioni tra matrici con moltiplicazioni eseguite sulla GPU

N = 1000 

# A and D need to be square and invertible

# B and C need to be conformable with A and D (BD, BDC, A-BDC, CA, CAB, D-CAB)
# that means: b_cols == d_rows, c_cols == a_rows

# (A_BDC) need to be invertible. If it is, then (D-CAB) is also invertible


def block(P):
    p_cols = np.shape(P)[1]
    p_rows = np.shape(P)[0]

    # Controllo che la matrice input sia quadrata e che sia almeno grande 2x2
    if p_cols != p_rows or p_cols == 0 or p_cols == 1:
        return

    i = 1
    while(True):
        A = P[0:i, 0:i]
        B = P[0:i, i:p_cols]
        C = P[i:p_rows, 0:i]
        D = P[i:p_rows, i:p_cols]
        if np.shape(A)[0] == np.shape(C)[1] and np.shape(B)[1] == np.shape(D)[0] and i != 1: 
            break
        i += 1 
    return A, B, C, D


# Calcolo dell'inversa
def inversa(P):
    if np.shape(P)[0] < 10:
        return inv(P)

    a, b, c, d = block(P)

    identity_matrix_a = np.eye(np.shape(b)[0], np.shape(b)[0])
    identity_matrix_d = np.eye(np.shape(b)[1], np.shape(b)[1])

    zero_matrix_b = np.zeros(np.shape(b))
    zero_matrix_c = np.zeros(np.shape(c))

    inversa_a = inversa(a)
    inversa_d = inversa(d)

    new_a = inv(a-b@inversa_d@c)
    new_d = inv(d-c@inversa_a@b)
    new_b = -b@inversa_d
    new_c = -c@inversa_a

    return np.block([[new_a, zero_matrix_b], [zero_matrix_c, new_d]])@np.block([[identity_matrix_a, new_b], [new_c, identity_matrix_d]])



if __name__ == "__main__":
    # Matrice iniziale
    P = np.random.randint(N, size=(N,N))
    #print(P, "\n")

    inversa = inversa(P)

    res = inversa@P

    frobenius_norm = np.sqrt(np.sum(res*res))

    print(f"Errore: {np.sqrt(N)-frobenius_norm}")
    #print(inversa(P))
    #print("Vera Inversa: \n", inv(P))





