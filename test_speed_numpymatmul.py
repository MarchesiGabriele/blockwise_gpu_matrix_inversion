import numpy as np
import time

np.__config__.show()
np.random.seed(0)

size = 4000 
A, B = np.random.rand(size, size), np.random.rand(size, size)

start = time.monotonic()
np.dot(A, B)
end = time.monotonic()

print(f"time: {end-start}s")
