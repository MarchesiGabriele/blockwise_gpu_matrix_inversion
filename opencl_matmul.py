import pyopencl as cl
import numpy as np
import os
import warnings

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0' 
warnings.filterwarnings("ignore")


# https://cnugteren.github.io/tutorial/pages/page4.html

def matmul(matrix1, matrix2, N):
    # OpenCL Setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)




    # Buffers
    mf = cl.mem_flags
    A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, matrix1.nbytes, hostbuf = matrix1)
    B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, matrix2.nbytes, hostbuf = matrix2)
    C = cl.Buffer(ctx, mf.WRITE_ONLY, matrix1.nbytes)

    # Kernels Creation
    prog = cl.Program(ctx,  """
                            __kernel void matmul(__global float* A, __global float* B, __global float* C, int N){
                                size_t i = get_global_id(1);
                                size_t j = get_global_id(0);
                                
                                float acc = 0.0f;

                                for(int c = 0; c<N; c++){
                                    acc += A[i*N + c] * B[j + c*N]; 
                                }
                                C[i*N + j] = acc;
                            }
                            """).build()

    # Kernel Execution
    pp = prog.matmul

    pp.set_args(A, B, C, np.int32(N))
    res = cl.enqueue_nd_range_kernel(queue, pp, [N,N], None)  # queue, kernel, global dims, local dims, offset
    queue.finish()

    
    # Lettura risultato finale
    cl.enqueue_copy(queue, matrix1, C)
    queue.finish()

    return matrix1



       



