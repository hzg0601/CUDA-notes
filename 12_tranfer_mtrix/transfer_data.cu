#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdint.h>
#include <assert.h>

#define N 3001
#define TILE_SIZE 32
#define cuda_error_check(r){\
    cudaError_t rr = r;\
    if(rr != cudaSuccess){\
        printf("Error %s at line %d in file %s\n", cudaGetErrorString(rr), __LINE__, __FILE__);\
        exit(1);\
    }\
}

__managed__ int array[N][N];
__managed__ int gpu_shared_result[N][N];
__managed__ int gpu_global_result[N][N];
__managed__ int cpu_result[N][N];

__global__ void matrix_transpose_shared(int A[N][N],int B[N][N]){
    // 定义block内的共享内存，用于存储block内的数据
    __shared__ int block_temp[TILE_SIZE][TILE_SIZE+1];
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    if (ix < N && iy < N){
        block_temp[threadIdx.y][threadIdx.x] = A[iy][ix];
    }
    __syncthreads();
    int tran_ix = threadIdx.y+blockDim.x*blockIdx.x;
    int tran_iy = threadIdx.x+blockDim.y*blockIdx.y;
    if (tran_ix < N && tran_iy < N){
        B[tran_ix][tran_iy] = block_temp[threadIdx.x][threadIdx.y];
    }
    __syncthreads();
    }


__global__ void matrix_transpose_global(int A[N][N], int B[N][N]){
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    if ( ix < N && iy < N){
        B[iy][ix] = A[ix][iy];
    }
}

void cpu_transpose_matrix(int A[N][N], int B[N][N]){
    for(int i=0; i<N;i++){
        for(int j=0;j<N;j++){
            B[i][j] = A[j][i];
        }
    }
}

double get_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return ((double)(tv.tv_sec+tv.tv_usec*0.000001));
}

void init_data(int A[N][N]){
    uint32_t seed = (uint32_t) time(NULL);
    srand(seed);
    for (int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            A[i][j] = rand();
        }
    }
}

int main(){

    int gd_s = (N+TILE_SIZE-1)/(TILE_SIZE);
    dim3 block_size_s(TILE_SIZE,TILE_SIZE);
    dim3 grid_size_s(gd_s,gd_s);

    int s_s = 16;
    int gd_g = (N+s_s-1)/s_s;
    dim3 block_size_g(s_s,s_s);
    dim3 grid_size_g(gd_g,gd_g);
    printf("start to execute...\n");

    init_data(array);

    cudaDeviceSynchronize();
    printf("start to call kernel..\n");
    double gpu_start_global = get_time();
    matrix_transpose_global<<<grid_size_g,block_size_g>>>(array,gpu_global_result);
    cuda_error_check(cudaGetLastError());
    double gpu_end_global = get_time();
    printf("start to execute shared memory transpose..\n");
    double gpu_start_shared = get_time();
    matrix_transpose_shared<<<grid_size_s,block_size_s>>>(array,gpu_shared_result);
    cuda_error_check(cudaGetLastError());
    double gpu_end_shared = get_time();
    printf("start to execute cpu transpose..\n");
    double cpu_start = get_time();
    cpu_transpose_matrix(array,cpu_result);
    double cpu_end = get_time();
    printf("start to compare result..\n");

    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if (
                (gpu_global_result[i][j] !=gpu_shared_result[i][j]) | \
                (gpu_shared_result[i][j] != cpu_result[i][j]) | \
                (gpu_global_result[i][j] != cpu_result[i][j])
                ){
                printf("test failed\n");
                exit(-1);
            }
        }
    }
    printf("test passed.\n");
    double cpu_elap = cpu_end - cpu_start;
    double gpu_shared_elap = gpu_end_shared - gpu_start_shared;
    double gpu_global_elap = gpu_end_global - gpu_start_global;
    printf("cpu:%f ms, gpu_shared:%f ms, gpu_global:%f ms\n",cpu_elap,gpu_shared_elap,gpu_global_elap);
    return 0;

}