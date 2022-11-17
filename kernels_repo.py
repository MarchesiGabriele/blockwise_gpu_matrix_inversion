# working kernel for square matrices with tiling where N is multiple of DIM
__kernel void matmul(__global float* A, __global float* B, __global float* C, int M, int K, int N){
    const int loc_row = get_local_id(1);
    const int loc_col = get_local_id(0);

    const int local_size = get_local_size(1);

    const int row = get_local_size(1)*get_group_id(1) + loc_row; 
    const int col = get_local_size(0)*get_group_id(0) + loc_col; 

    __local float Asub[16][16];
    __local float Bsub[16][16];

    float acc = 0.0f;

    const short numTiles = (int)ceil((float)K/local_size);

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
        Asub[loc_row][loc_col] = A[row*K + loc_col + i*local_size];
        Bsub[loc_row][loc_col] = B[loc_row*N + col + i*local_size*N];

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int c = 0; c<local_size; c++){
            acc += Asub[loc_row][c]*Bsub[c][loc_col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[row*N + col] = acc;


# Working kernel for rectangular matrix and tiling, good performance:w

__kernel void matmul(__global float* A, __global float* B, __global float* C, int M, int K, int N){
    const int loc_row = get_local_id(1);
    const int loc_col = get_local_id(0);

    const int local_size = get_local_size(1);

    const int row = get_local_size(1)*get_group_id(1) + loc_row; 
    const int col = get_local_size(0)*get_group_id(0) + loc_col; 

    __local float Asub[16][16];
    __local float Bsub[16][16];

    float acc = 0.0f;

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
        if(loc_col+i*local_size < K && row < M){
            Asub[loc_row][loc_col] = A[row*K + loc_col + i*local_size];
        }else{
            Asub[loc_row][loc_col] = 0.0f; 
        }

        if(i*local_size+loc_row < K && col < N){
            Bsub[loc_row][loc_col] = B[loc_row*N + col + i*local_size*N];
        }else{
            Bsub[loc_row][loc_col] = 0.0f; 
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for(int c = 0; c<local_size; c++){
            acc += Asub[loc_row][c]*Bsub[c][loc_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if(row < M && col < N){
        C[row*N + col] = acc;
    }
    
}


