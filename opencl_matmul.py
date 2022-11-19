import pyopencl as cl
import numpy as np
import os
import warnings
from time import perf_counter as pt 
import math

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0' 
warnings.filterwarnings("ignore")

DIM = 16 

# https://cnugteren.github.io/tutorial/pages/page4.html
#TODO: spostare creazione context fuori da questa funzione e metterlo dentro la funzione main, oppure dentro la prima chiamata della funzione inversa (per crearlo una sola volta)

def matmul(matrix1, matrix2, M, K, N, fp32, ctx, queue):

    # OpenCL Setup
    #ctx = cl.create_some_context()
    #queue = cl.CommandQueue(ctx)

    # Buffers
    start = pt()
    if fp32: 
        out_matrix = np.random.rand(M,N).astype(np.float32)

    else:
        out_matrix = np.random.rand(M,N)
    end = pt()

    #print(f"Tempo creazione matrice: {end-start}")

    mf = cl.mem_flags
    A = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = matrix1)
    B = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf = matrix2)
    C = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf = out_matrix)

    # Kernels Creation
    # NB: max local mem size: 65536 byte (for each workgroup) == 90 FP64 values
    # M = altezza matrice A, N = larghezza matrice B, K = larghezza matrice A = altezza matrice B
    #TODO: fare la transposta della matrice globale B
    if fp32:
        prog = cl.Program(ctx,  """

                            #define WPT 4
                            __kernel void matmul(__global float* A, __global float* B, __global float* C, int M, int K, int N){
                                const int loc_row = get_local_id(1);
                                const int loc_col = get_local_id(0);

                                const int local_size = get_local_size(1);

                                const int row = get_local_size(1)*get_group_id(1) + loc_row; 
                                const int col = get_local_size(0)*get_group_id(0) + loc_col; 

                                __local float Asub[16][16];
                                __local float Bsub[16][16];

                                float acc[WPT];
                                for (int w=0; w<WPT; w++) {
                                    acc[w] = 0.0f;
                                }

                                const short numTiles = (short)ceil((float)K/local_size);
    
                                /*
                                Asub:
                                i*local_size = indica a quale tile ci troviamo.
                                row*K = indica la riga della matrice A su cui mi trovo.
                                loc_col = scorro orizzontalmente la matrice A, DIM elementi alla volta. 

                                Bsub: 
                                col = indica la colonna della matrice B su cui ci troviamo attualmente
                                loc_row*N = indica la riga della matrice B su cui ci troviamo
                                i*local_size*N = indica a quale tile mi trovo
                                */

                                for(int i = 0; i<numTiles; i++){
                                    if(loc_col*WPT+3+i*local_size < K && row < M){
                                        Asub[loc_row][loc_col*WPT] = A[row*K + loc_col*WPT + i*local_size];
                                        Asub[loc_row][loc_col*WPT+1] = A[row*K + loc_col*WPT +1 + i*local_size];
                                        Asub[loc_row][loc_col*WPT+2] = A[row*K + loc_col*WPT +2 + i*local_size];
                                        Asub[loc_row][loc_col*WPT+3] = A[row*K + loc_col*WPT +3 + i*local_size];
                                    }
                                    else if(loc_col*WPT+2+i*local_size < K && row < M){
                                        Asub[loc_row][loc_col*WPT] = A[row*K + loc_col*WPT + i*local_size];
                                        Asub[loc_row][loc_col*WPT+1] = A[row*K + loc_col*WPT +1 + i*local_size];
                                        Asub[loc_row][loc_col*WPT+2] = A[row*K + loc_col*WPT +2 + i*local_size];
                                        Asub[loc_row][loc_col*WPT+3] = 0.0f; 
                                    }
                                    else if(loc_col*WPT+1+i*local_size < K && row < M){
                                        Asub[loc_row][loc_col*WPT] = A[row*K + loc_col*WPT + i*local_size];
                                        Asub[loc_row][loc_col*WPT+1] = A[row*K + loc_col*WPT +1 + i*local_size];
                                        Asub[loc_row][loc_col*WPT+2] = 0.0f; 
                                        Asub[loc_row][loc_col*WPT+3] = 0.0f; 
                                     
                                    }
                                    else if(loc_col*WPT+i*local_size < K && row < M){
                                        Asub[loc_row][loc_col*WPT] = A[row*K + loc_col*WPT + i*local_size];
                                        Asub[loc_row][loc_col*WPT+1] = 0.0f; 
                                        Asub[loc_row][loc_col*WPT+2] = 0.0f; 
                                        Asub[loc_row][loc_col*WPT+3] = 0.0f; 
 
                                    }
                                    else{
                                        Asub[loc_row][loc_col*WPT] = 0.0f; 
                                        Asub[loc_row][loc_col*WPT+1] = 0.0f; 
                                        Asub[loc_row][loc_col*WPT+2] = 0.0f; 
                                        Asub[loc_row][loc_col*WPT+3] = 0.0f; 
                                    }

                                    if(i*local_size+loc_row < K && col*WPT+3 < N){
                                        Bsub[loc_row][loc_col*WPT] = B[loc_row*N + col*WPT + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+1] = B[loc_row*N + col*WPT+1 + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+2] = B[loc_row*N + col*WPT+2 + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+3] = B[loc_row*N + col*WPT+3 + i*local_size*N];
                                    }
                                    else if(i*local_size+loc_row < K && col*WPT+2 < N){
                                        Bsub[loc_row][loc_col*WPT] = B[loc_row*N + col + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+1] = B[loc_row*N + col+1 + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+2] = B[loc_row*N + col+2 + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+3] = 0.0f; 
                                    }
                                    else if(i*local_size+loc_row < K && col*WPT+1 < N){
                                        Bsub[loc_row][loc_col*WPT] = B[loc_row*N + col + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+1] = B[loc_row*N + col+1 + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+2] = 0.0f; 
                                        Bsub[loc_row][loc_col*WPT+3] = 0.0f; 
                                    }
                                    else if(i*local_size+loc_row < K && col*WPT < N){
                                        Bsub[loc_row][loc_col*WPT] = B[loc_row*N + col + i*local_size*N];
                                        Bsub[loc_row][loc_col*WPT+1] = 0.0f;  
                                        Bsub[loc_row][loc_col*WPT+2] = 0.0f; 
                                        Bsub[loc_row][loc_col*WPT+3] = 0.0f; 
                                    }
                                    else{
                                        Bsub[loc_row][loc_col*WPT] = 0.0f; 
                                        Bsub[loc_row][loc_col*WPT+1] = 0.0f;  
                                        Bsub[loc_row][loc_col*WPT+2] = 0.0f; 
                                        Bsub[loc_row][loc_col*WPT+3] = 0.0f; 
                                    }

                                    barrier(CLK_LOCAL_MEM_FENCE);
                                    
                                    for(int c = 0; c<local_size; c++){
                                        for(int Tc = 0; c<local_size; c++){
                                        acc[0] += Asub[loc_row][c]*Bsub[c][loc_col*WPT];
                                        acc[1] += Asub[loc_row][c]*Bsub[c][loc_col*WPT+1];
                                        acc[2] += Asub[loc_row][c]*Bsub[c][loc_col*WPT+2];
                                        acc[3] += Asub[loc_row][c]*Bsub[c][loc_col*WPT+3];
                                        }
                                    }

                                    barrier(CLK_LOCAL_MEM_FENCE);
                                }

                                if(row < M && col*WPT+3 < N){
                                    C[row*N + col*WPT] = acc[0];
                                    C[row*N + col*WPT+1] = acc[1];
                                    C[row*N + col*WPT+2] = acc[2];
                                    C[row*N + col*WPT+3] = acc[3];
                                }
                                else if(row < M && col*WPT+3 < N){
                                    C[row*N + col*WPT] = acc[0];
                                    C[row*N + col*WPT+1] = acc[1];
                                    C[row*N + col*WPT+2] = acc[2];
                                }
                                else if(row < M && col*WPT+3 < N){
                                    C[row*N + col*WPT] = acc[0];
                                    C[row*N + col*WPT+1] = acc[1];
                                }
                                else if(row < M && col*WPT+3 < N){
                                    C[row*N + col*WPT] = acc[0];
                                }
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
    if M%DIM != 0:
        offset_M = M + DIM - (M%DIM) 
    else:
        offset_M = M 

    if N%DIM != 0:
        offset_N = N + DIM - (N%DIM) 
    else: 
        offset_N = N 

    print(offset_M)
    print(offset_N)

    pp.set_args(A, B, C, np.int32(M), np.int32(K), np.int32(N))
    st = pt() 
    print(math.ceil(offset_N/4))
    res = cl.enqueue_nd_range_kernel(queue, pp, [np.int32(math.ceil(offset_N/4)), offset_M], [np.int32(DIM/4), DIM], None)  # queue, kernel, global dims, local dims, offset
    queue.finish()
    end = pt() 


    print(f"\nTempo Computazione: {end-st}")
    print(f"FLOPS Computazione: {(M*K*2*N)/((end-st)*1e9)} GFLOPS")
    
    # Lettura risultato finale
    cl.enqueue_copy(queue, out_matrix, C)
    queue.finish()

    return out_matrix 


