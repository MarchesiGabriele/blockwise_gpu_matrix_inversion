import numpy as np
from numpy.linalg import inv

LIM = 1000

# A and D need to be square and invertible
# B and C need to be conformable with A and D (BD, BDC, A-BDC, CA, CAB, D-CAB)
# that means: b_cols == d_rows, c_cols == a_rows
# (A_BDC) need to be invertible. If it is, then (D-CAB) is also invertible


def block(P):
    p_cols = np.shape(P)[1]
    p_rows = np.shape(P)[0]

    print("blocking...")

    # Controllo che la matrice input sia quadrata e che sia almeno grande 2x2
    if p_cols != p_rows or p_cols == 0 or p_cols == 1:
        return
    
    idx = p_cols//2

    A = P[0:idx, 0:idx]
    B = P[0:idx, idx:p_cols+1]
    C = P[idx:p_rows+1, 0:idx]
    D = P[idx:p_rows+1, idx:p_cols+1]

    """
    print(P, "\n")
    print(A,"\n")
    print(B,"\n")
    print(C,"\n")
    print(D,"\n")
    """
    return A, B, C, D


# Calcolo dell'inversa
def inversa(P):
    if np.shape(P)[0] <= LIM:
        return inv(P)

    a, b, c, d = block(P)

    print("backing...")

    identity_matrix_a = np.eye(np.shape(b)[0], np.shape(b)[0])
    identity_matrix_d = np.eye(np.shape(b)[1], np.shape(b)[1])

    zero_matrix_b = np.zeros(np.shape(b))
    zero_matrix_c = np.zeros(np.shape(c))

    inversa_a = inversa(a)
    inversa_d = inversa(d)
    
    #TODO: usare inv() oppure inversa() ???
    new_a = inversa(a-b@inversa_d@c)
    new_d = inversa(d-c@inversa_a@b)
    new_b = -b@inversa_d
    new_c = -c@inversa_a

    return np.block([[new_a, zero_matrix_b], [zero_matrix_c, new_d]])@np.block([[identity_matrix_a, new_b], [new_c, identity_matrix_d]])



