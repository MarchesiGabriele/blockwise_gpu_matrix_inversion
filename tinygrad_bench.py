from tinygrad.tensor import Tensor
import time

N = 8192 

a,b = Tensor.rand(N,N), Tensor.rand(N,N)

start = time.time()
c = (a.reshape(N,1,N) * b.T.reshape(1,N,N)).sum(axis=2)
c.realize()
end = time.time()

print((c.numpy() - a.numpy() @ b.numpy()).mean())
print("Time: ", end-start, " s")


