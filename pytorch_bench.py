import torch
import time

N = 4096

A,B = torch.rand([N,N], dtype=torch.float32, device="cuda"), torch.rand([N,N], dtype=torch.float32, device="cuda")
start = time.time()
C = torch.mm(A, B)  # Moltiplicazione su GPU
end = time.time()

print("Time: ", (end-start), "s")
print("GFLOPS: ", (N**3 * 2)/((end-start)*10**9), " GFLOPS")

