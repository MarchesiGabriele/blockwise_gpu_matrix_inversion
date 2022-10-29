import pyopencl as cl
import numpy as np


# https://cnugteren.github.io/tutorial/pages/page4.html


def matmul(matrix1, matrix2, N):
    # OpenCL Setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)




    # Buffers
    mf = cl.mem_flags
    A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_POINTER, matrix1.nbytes, hostbuf = matrix1)
    B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_POINTER, matrix2.nbytes, hostbuf = matrix2)
    C = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_POINTER, matrix1.nbytes)

    # Kernels Creation
    prog = cl.Program(ctx,  """
                            """).build()

    # Kernel Execution
    pp = prog.matmul

    pp.set_args(A, B, C, np.int32(N))
    res = cl.enqueue_nd_range_kernel(queue, pp, N)  # queue, kernel, global dims, local dims, offset
    queue.finish()

    
    # Lettura risultato finale
    cl.enqueue_copy(queue, matrix1, C)
    queue.finish()

    return matrix1



       



