import numpy as np

N = 5 


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
    return A, B, C, D, P

def inversa(P):
    a, b, c, d, p = block(P)



    print(a)
    print(b)
    print(c)
    print(d)



if __name__ == "__main__":
    # Matrice iniziale
    P = np.random.randint(N, size=(N,N))
    print(P, "\n")

    print(inversa(P))


