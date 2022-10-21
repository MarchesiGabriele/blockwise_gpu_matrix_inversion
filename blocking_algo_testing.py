import numpy as np

N = 5 


# A and D need to be square and invertible

# B and C need to be conformable with A and D (BD, BDC, A-BDC, CA, CAB, D-CAB)
# that means: b_cols == d_rows, c_cols == a_rows

# (A_BDC) need to be invertible. If it is, then (D-CAB) is also invertible



def block():
    # Matrice iniziale
    P = np.random.randint(N, size=(N,N))

    print(P, "\n")

    p_cols = np.shape(P)[1]
    p_rows = np.shape(P)[0]

    i = 1
    while(True):
        A = P[0:i, 0:i]
        B = P[0:i, i:p_cols]
        C = P[i:p_rows, 0:i]
        D = P[i:p_rows, i:p_cols]
        if np.shape(A)[0] == np.shape(C)[1] and np.shape(B)[1] == np.shape(D)[0] and i != 1: 
            break
        i += 1 

    print(A)
    print(B)
    print(C)
    print(D)



if __name__ == "__main__":
    block()

