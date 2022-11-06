import pyopencl as cl
import numpy as np
import os
import warnings

os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '0' 
warnings.filterwarnings("ignore")

DIM = 16 


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

                                    size_t loc_row = get_local_id(0);
                                    size_t loc_col = get_local_id(1);

                                    int local_size = get_local_size(1);

                                    int globalRow = local_size*get_group_id(0) + loc_row; 
                                    int globalCol = local_size*get_group_id(1) + loc_col; 


                                    __local float Asub[256];
                                    __local float Bsub[256];

                                    float acc = 0.0f;
        
                                    // calcolo numero di tiles
                                    int numTiles;
                                    int remainingTile;
                                    
                                    // Controllo se le tiles coprono tutte le matrici o no 
                                    if((K%local_size) == 0){
                                        numTiles = K/local_size;
                                        remainingTile = 0;
                                    }else{
                                        numTiles = (int)ceil((float)K/local_size);
                                        remainingTile = K%local_size;
                                    }

                                    if(col == 0 && row == 0) 
                                        printf("numtiles: %i, remTiles: %i", numTiles, remainingTile);

                                    for(int i = 0; i<numTiles; i++){
                                        //printf("Global: %i %i        Local: %i %i       value: %f ", globalRow, globalCol, loc_row, loc_col, A[loc_col*M + globalRow + i*local_size*M]);
                                        // Asub e Bsub sono le trasposte rispetto ai valori nella matrice iniziale 
                                        Asub[loc_col*local_size + loc_row] = A[globalRow*K + loc_col + i*local_size];
                                        Bsub[loc_col*local_size + loc_row] = B[loc_row*N + globalCol + i*local_size*N];

                                        barrier(CLK_LOCAL_MEM_FENCE);
                                        
                                        /*if(row == 3 && col == 3){
                                            for(int ss = 0; ss<local_size*local_size; ss++){
                                                printf("%f %i", Asub[ss], i);
                                            }
                                            printf("stop");
                                        } */
 
                                        if(i != numTiles-1 || remainingTile == 0){
                                            for(int c = 0; c<local_size; c++){
                                                // Il prodotto tra B' e A' permette di avere il risultato con gli index nella posizione corretta 
                                                acc += Bsub[c + loc_col*local_size] * Asub[c*local_size + loc_row];
                                                //printf("%f %f", Asub[c + loc_col*local_size] , Bsub[c*local_size + loc_row]);
                                            }
                                        }
                                        else{
                                            for(int c = 0; c<remainingTile; c++){
                                                acc += Bsub[c + loc_col*local_size] * Asub[c*local_size + loc_row];
                                            }

                                        }

                                        barrier(CLK_LOCAL_MEM_FENCE);
                                    }
            
                                    //printf("%f     row %i, col %i", acc, globalRow, globalCol);
                                    if(globalRow < M && globalCol < N){
                                        C[globalRow*N + globalCol] = acc;
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

    offset_M = M + (M%DIM)
    offset_N = N + (N%DIM)

    pp.set_args(A, B, C, np.int32(M), np.int32(K), np.int32(N))
    res = cl.enqueue_nd_range_kernel(queue, pp, [M+offset_M, N+offset_N], [DIM, DIM], None)  # queue, kernel, global dims, local dims, offset
    queue.finish()

    
    # Lettura risultato finale
    cl.enqueue_copy(queue, out_matrix, C)
    queue.finish()

    return out_matrix 



       



