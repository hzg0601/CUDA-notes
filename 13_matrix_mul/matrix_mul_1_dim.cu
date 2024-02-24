#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 32
#define M 40
#define N 50
#define K 60

__global__ void gpu_mul_global(int *A, int *B,int *C, int m, int n, int k){
    // A m*k , B k*n, C m*n
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int temp = 0;
    if (iy < m && ix < n){
        for(int i=0;i<k;i++){
            temp += A[iy*k+i] * B[n*i+ix];
        }
        C[iy*n + ix] = temp;
    }
}

__global__ void gpu_mul_shared(int *A, int *B, int *C, int m, int n, int k){
    // A m*k , B k*n, C m*n
    __shared__ int A_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int B_temp[BLOCK_SIZE][BLOCK_SIZE];
    int temp =0;
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    for(int i=0; i< (k+BLOCK_SIZE-1)/BLOCK_SIZE; i++){
        if (iy < m && i*BLOCK_SIZE + threadIdx.x < k){
            A_temp[threadIdx.y][threadIdx.x] = A[iy*k+i*BLOCK_SIZE + threadIdx.x];
        }
        else{
            A_temp[threadIdx.y][threadIdx.x] = 0;
        }
        if (ix < n && i*BLOCK_SIZE + threadIdx.y < k){
            B_temp[threadIdx.y][threadIdx.x] = B[ix+(i*BLOCK_SIZE+threadIdx.y)*n];
        }
        else{
            B_temp[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        //注意此处为逐BLOCK计算和
        for(int j=0; j<BLOCK_SIZE;j++){
            temp += A_temp[threadIdx.y][j] * B_temp[j][threadIdx.x];
        }
        __syncthreads();
    }
    // 注意乘以的是数据的宽度
    if(iy<m && ix <n){
        C[iy*n + ix] = temp;
    }
}

void cpu_mul(int *A, int *B, int *C, int m,int n, int k){
    // A m*k , B k*n, C m*n
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            int temp = 0;
            for (int t=0; t<k;t++){
                temp += A[i*k+t] * B[t*n + j];
            }
            C[i*n + j] = temp;
        }
    }
}

void init_data(int * arr, int x, int y){
    uint32_t seed = (uint32_t) time(NULL);
    srand(seed);
    for (int i=0; i<x;i++){
        for (int j=0; j<y;j++){
            arr[i*y+j] = rand();
        }
    }
}

double get_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double _time = (double)(tv.tv_sec+tv.tv_usec*0.000001);
    return _time;
}


int main(){
    //初始化数据
    int *host_A, *host_B, *host_C_g, *host_C_s, *host_C_c;
    host_A = (int *)malloc(sizeof(int) * M * K);
    host_B = (int *)malloc(sizeof(int) * K * N);
    host_C_g = (int *)malloc(sizeof(int) * M * N);
    host_C_s = (int *)malloc(sizeof(int) * M * N);
    host_C_c = (int *)malloc(sizeof(int) * M * N);
    init_data(host_A, M, K);
    init_data(host_B, K, N);
    memset(host_C_g, 0, sizeof(int)*M*N);
    memset(host_C_s, 0, sizeof(int)*M*N);
    memset(host_C_c, 0, sizeof(int)*M*N);

    int *device_A, *device_B, *device_C_g, *device_C_s;
    cudaMalloc((int **)&device_A,sizeof(int)*M*K);
    cudaMalloc((int **)&device_B,sizeof(int)*K*N);
    cudaMalloc((int **)&device_C_g,sizeof(int)*M*N);
    cudaMalloc((int **)&device_C_s,sizeof(int)*M*N);

    //转移数据
    cudaMemcpy(device_A,host_A,sizeof(int)*M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(device_B,host_B,sizeof(int)*K*N,cudaMemcpyHostToDevice);
    cudaMemcpy(device_C_g,host_C_g,sizeof(int)*M*N,cudaMemcpyHostToDevice);
    cudaMemcpy(device_C_s,host_C_s,sizeof(int)*M*N,cudaMemcpyHostToDevice);
    // call kernel funciton
    dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
    //! 注意此处的N和M的位置
    dim3 grid_size((N+BLOCK_SIZE-1)/BLOCK_SIZE,(M+BLOCK_SIZE-1)/BLOCK_SIZE);

    double global_start = get_time();
    gpu_mul_global<<<grid_size,block_size>>>(device_A,device_B,device_C_g,M,N,K);
    cudaMemcpy(host_C_g, device_C_g,sizeof(int)*M*N,cudaMemcpyDeviceToHost);
    double global_end = get_time();

    double shared_start = get_time();
    gpu_mul_shared<<<grid_size,block_size>>>(device_A,device_B,device_C_s,M,N,K);
    cudaMemcpy(host_C_s,device_C_s,sizeof(int)*M*N,cudaMemcpyDeviceToHost);
    double shared_end = get_time();

    double cpu_start = get_time();
    cpu_mul(host_A, host_B, host_C_c, M, N, K);
    double cpu_end = get_time();

    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if(
                (host_C_g[i*N+j] != host_C_s[i*N+j]) |\
                (host_C_g[i*N+j] != host_C_c[i*N+j]) |\
                (host_C_s[i*N+j] != host_C_c[i*N+j])
             ){
                printf("test failed\n");
                free(host_A);
                free(host_B);
                free(host_C_g);
                free(host_C_s);
                free(host_C_c);

                cudaFree(device_A);
                cudaFree(device_B);
                cudaFree(device_C_g);
                cudaFree(device_C_s);
                exit(-1);
             }

        }
    }
    printf("test pass\n");
    double global_elap = global_end - global_start;
    double shared_elap = shared_end - shared_start;
    double cpu_elap = cpu_end - cpu_start;
    //二维 gpu global time elapse:0.091566ms,gpu shared time elapse:0.000002ms, cpu time elapse:0.001038ms
    //一维 gpu global time elapse:0.000027ms,gpu shared time elapse:0.000014ms, cpu time elapse:0.000184ms
    
    //gpu global time elapse:0.000027ms,gpu shared time elapse:0.000018ms, cpu time elapse:0.000250ms
    //gpu global :0.107824,gpu shared :0.07251
    printf("gpu global time elapse:%fms,gpu shared time elapse:%fms, cpu time elapse:%fms\n",global_elap, shared_elap, cpu_elap);
    printf("gpu global :%f,gpu shared :%f\n",global_elap/cpu_elap, shared_elap/cpu_elap);
    return 0;


    free(host_A);
    free(host_B);
    free(host_C_g);
    free(host_C_s);
    free(host_C_c);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C_g);
    cudaFree(device_C_s);
    return 0;

}