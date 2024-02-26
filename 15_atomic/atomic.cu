// 利用原子操作直接求数组的和
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define BLOCK_SIZE 256
#define N 1000000
#define GRID_SIZE ((N+BLOCK_SIZE-1)/BLOCK_SIZE)

__managed__ int arr[N];
__managed__ int output[1] = {0};


__global__ void gpu_add_atomic(int *A ,int num, int *output){
    // shared变量不允许初始化；
    __shared__ int block_temp[BLOCK_SIZE] ;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int temp = 0;
    //首先将每个idx相同的A中的元素相加，缓存到共享内存变量中
    for(int i=idx; i< num;i+=blockDim.x * gridDim.x){
         temp+= A[i];
    }
    block_temp[threadIdx.x] = temp;
    __syncthreads();
    //执行block内的inplace加法,执行完毕后每个共享变量的0位置都是该block内和；
    // !!!注意共享变量操作不能同时做左值和右值
    for (int i=BLOCK_SIZE/2;i>=1;i/=2){
        int temp = 0;
        if (threadIdx.x< i){
            temp = block_temp[threadIdx.x] + block_temp[threadIdx.x+i];
        }
        __syncthreads();
        if(threadIdx.x < i){
            block_temp[threadIdx.x] = temp;
        }
        __syncthreads();
    }

    if (threadIdx.x==0 && blockDim.x * blockIdx.x < num) atomicAdd(output, block_temp[0]);
}


void cpu_add(int *A, int num, int * output){
    for(int i=0;i<num;i++){
        output[0] += A[i];
    }
}

double get_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double _time = (double)(tv.tv_usec*0.000001 + tv.tv_sec);
    return _time;
}


void init_data(int *A, int num){
    srand(666);
    for (int i=0;i<num;i++){
        A[i] = rand();
    }
}


int main(){
    printf("start.\n");

    cudaDeviceSynchronize();
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float gpu_elapse;
    int cpu_output[1] = {0};
    init_data(arr,N);
    printf("create event and call kernel\n");
    cudaEventRecord(start);
    double gs,ge,gp;
    gs = get_time();
    gpu_add_atomic<<<GRID_SIZE,BLOCK_SIZE>>>(arr,N,output);
    ge = get_time();
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpu_elapse,start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    printf("start cpu function\n");

    double cpu_start, cpu_end, cpu_elapse;
    cpu_start = get_time();
    cpu_add(arr,N,cpu_output);
    cpu_end = get_time();
    cpu_elapse = cpu_end - cpu_start;
    gp = ge-gs;
    printf("gpu record elapse:%.11f ms, gp full time:%f, cpu elapse:%.11f ms.\n",gpu_elapse, gp, cpu_elapse);
    printf("gpu result:%d, cpu result:%d.\n",output[0],cpu_output[0]);
    return 0;


}