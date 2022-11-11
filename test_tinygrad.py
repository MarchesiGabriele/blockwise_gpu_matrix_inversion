from tinygrad.tensor import Tensor
import time

N = 10

A

x = Tensor.eye(3, requires_grad=True)
y = Tensor([[2.0,0,-2.0]], requires_grad=True)

print(x)
print()
print(y)



start = time.monotonic()
z = y.matmul(x)
end = time.monotonic()


print(f"time: {end-start}s")
