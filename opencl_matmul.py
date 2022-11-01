import pyopencl as cl
import numpy as np
import os
import warnings

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0' 
warnings.filterwarnings("ignore")


# https://cnugteren.github.io/tutorial/pages/page4.html
#TODO: spostare creazione context fuori da questa funzione e metterlo dentro la funzione main, oppure dentro la prima chiamata della funzione inversa (per crearlo una sola volta)

def matmul(matrix1, matrix2, M, K, N, fp32):

    # OpenCL Setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Buffers
    if fp32: 
        out_matrix = np.random.rand(M,N).astype(np.float32)
    else:
        out_matrix = np.random.rand(M,N)

    mf = cl.mem_flags
    A = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = matrix1)
    B = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = matrix2)
    C = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf = out_matrix)

    # Kernels Creation
    # NB: max local mem size: 65536 byte (for each workgroup) == 90 FP64 values
    # M = altezza matrice A, N = larghezza matrice B, K = larghezza matrice A = altezza matrice B
    if fp32:
        prog = cl.Program(ctx,  """
                                __kernel void matmul(__global float* A, __global float* B, __global float* C, int M, int K, int N){
                                    size_t row = get_global_id(0);
                                    size_t col = get_global_id(1);

                                    float acc = 0.0f;
                                    for(int c = 0; c<K; c++){
                                        acc += A[row*K + c] * B[col + c*N]; 
                                    }
                                    C[row*N + col] = acc;
                                }
                                """).build()
    else:
        prog = cl.Program(ctx,  """
                                #pragma OPENCL EXTENSION cl_khr_fp64 : enable(res)
                                __kernel void matmul(__global double* A, __global double* B, __global double* C, int M, int K, int N){
                                    size_t row = get_global_id(0);
                                    size_t col = get_global_id(1);

                                    double acc = 0.0;
                                    for(int c = 0; c<K; c++){
                                        acc += A[row*K + c] * B[col + c*N]; 
                                    }
                                    C[row*N + col] = acc;
                                }
                                """).build()

        
    # Kernel Execution
    pp = prog.matmul

    pp.set_args(A, B, C, np.int32(M), np.int32(K), np.int32(N))
    res = cl.enqueue_nd_range_kernel(queue, pp, [M, N], None, None)  # queue, kernel, global dims, local dims, offset
    queue.finish()

    
    # Lettura risultato finale
    cl.enqueue_copy(queue, out_matrix, C)
    queue.finish()

    return out_matrix 



       



