#include <cuda_runtime.h>
#include <iostream>
#include "../tools/common.cuh"
#define SHARED_LEN 32
#define ELMENT_LEN 64
// extern __shared__ float s_ar[];

__global__ void kernel_dynamic(float *A, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + bid * blockDim.x;
    // __shared__ float s_ar[]; # 不能在核函数内部声明动态共享内存
    extern __shared__ float s_ar[];
    if (gid < N) s_ar[tid] = A[gid];
    __syncthreads();
    if (tid == 0){
        for (int i=0;i<SHARED_LEN;i++){
            printf("bid, tid, gid, s_ar[i]: %d,%d,%d,%.2f\n",bid,tid,gid,s_ar[i]);
        }
    }
}

void init_data(float *A, const int N){
    for (int i=0;i<N;i++) A[i] = (float) (rand()%10); //生成0-9之间的随机数
}

int main(int argc, char *argv[]){

    set_device_by_arg(argc,argv);

    float * host_A;
    int size_bytes = ELMENT_LEN * sizeof(float);
    host_A = (float *)malloc(size_bytes);
    init_data(host_A,ELMENT_LEN);

    float *device_A;
    cudaMalloc((float **)&device_A, size_bytes);
    cudaMemset(device_A, 0, size_bytes);

    cudaMemcpy(device_A, host_A, size_bytes, cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid(2);
    kernel_dynamic<<<grid,block,32>>>(device_A, ELMENT_LEN);

    cudaDeviceSynchronize();

    cudaMemcpy(host_A, device_A, size_bytes, cudaMemcpyDeviceToHost);

    free(host_A);
    cudaFree(device_A);
    cudaDeviceReset();
    return 0;

}