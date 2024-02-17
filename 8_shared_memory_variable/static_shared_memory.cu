// #include <stdio.h>
// #include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include "../tools/common.cuh"
using namespace std;

__global__ void kernel_static(float *A, const int N){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float s_ar[32];

    if (gid < N){
        s_ar[tid] = A[gid];
    }
    __syncthreads();
    //每个线程块内部的tid==0的点，确定是不同的线程块
    if (tid == 0){
        for (int i=0; i<32;++i){
            printf("kernel_static:%f, blockIdx:%d,blockDim:%d\n",s_ar[i],bid,blockDim.x);
        }
    }
}

void init_data(float * A, const int N){
    for (int i=0;i<N;i++){
        A[i] = (float) i;
    }
}

int main(int argc, char * argv[]){
    //设置GPU
    set_device_by_arg(argc, argv);
    //定义元素大小
    int count = 64;
    int size_bytes = count * sizeof(float);

    //定义主机内存
    float * host_A;
    host_A = (float *)malloc(size_bytes);
    // memset(host_A, 0.f, size_bytes);
    init_data(host_A, count);
    // for (int i=0;i<64;i++) {
    //     printf("the %d of host A is: %f", i, host_A[i]);
    // }
    //定义设备内存

    float * device_A;
    cudaMalloc((float **)&device_A,size_bytes);
    cudaMemset(device_A, 0.f, size_bytes);

    //定义核函数的block,grid
    dim3 block(32);
    dim3 grid(2);

    //数据转移至设备；
    cudaMemcpy(device_A, host_A, size_bytes, cudaMemcpyHostToDevice);
    kernel_static<<<grid,block>>>(device_A, count);
    //同步
    cudaDeviceSynchronize();
    //数据回传
    cudaMemcpy(host_A, device_A, size_bytes, cudaMemcpyDeviceToHost);

    free(host_A);
    cudaFree(device_A);
    cudaDeviceReset();
    cout << "done" <<endl;
    return 0;

}

