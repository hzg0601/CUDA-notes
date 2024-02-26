#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define BLOCK_SIZE 256
#define GRID_SIZE ((N+BLOCK_SIZE-1)/BLOCK_SIZE)
#define N (1024*1024)
#define FULL_DATA_SIZE (N*20)

__global__ void kernel(int *a,int *b, int *c, int n){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if(idx < n){
        int idx1 = (idx+1)%256;
        int idx2 = (idx+2)%256;
        float as = (a[idx] + a[idx1] + a[idx2])/3.0f;
        float bs = (b[idx] + b[idx1] + b[idx2])/3.0f;
        c[idx] = (as + bs)/2.0f;
    }
}

void init_data(int * a, int n){
    srand(666);
    for(int i=0;i<n;i++) a[i] = rand();
}


int main(){
    // 检查设备是否支持deviceOverlap属性
    cudaDeviceProp prop;
    int device_id;
    cudaGetDevice(&device_id);
    cudaGetDeviceProperties(&prop, device_id);
    if (!prop.deviceOverlap){
        printf("device id %d do not support deviceOverlap\n",device_id);
        exit(-1);
    }

    // 定义host变量,分配host锁页内存，初始化
    int * host_a, *host_b, *host_c;
    // host_a = (int *)malloc(FULL_DATA_SIZE * sizeof(int));
    // host_b = (int *)malloc(FULL_DATA_SIZE * sizeof(int));
    // host_c = (int *)malloc(FULL_DATA_SIZE * sizeof(int));
    // 使用stream需使用cudaHostAlloc函数定义host locked memory
    cudaHostAlloc((int **)&host_a, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((int **)&host_b, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((int **)&host_c, FULL_DATA_SIZE*sizeof(int), cudaHostAllocDefault);

    init_data(host_a,FULL_DATA_SIZE);
    init_data(host_b,FULL_DATA_SIZE);
    // 定义各个流的device变量，分配内存；
    int *dev_a0,*dev_a1,*dev_b0,*dev_b1,*dev_c0,*dev_c1;
    cudaMalloc((int **)&dev_a0, sizeof(int)*N);
    cudaMalloc((int **)&dev_a1, sizeof(int)*N);
    cudaMalloc((int **)&dev_b0, sizeof(int)*N);
    cudaMalloc((int **)&dev_b1, sizeof(int)*N);
    cudaMalloc((int **)&dev_c0, sizeof(int)*N);
    cudaMalloc((int **)&dev_c1, sizeof(int)*N);
    //声明并创建cudaEvent_t和cudaStream_t；
    cudaEvent_t start, end;
    cudaStream_t stream0,stream1;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // 定义BLOCK和GRID
    dim3 block_size(BLOCK_SIZE);
    dim3 grid_size(GRID_SIZE);

    cudaEventRecord(start);
    // 运行10次kernel,stream0和stream1交替执行,host_a占据前半部分,host_b占据后半部分，长度均为N
    for(int i=0; i<FULL_DATA_SIZE;i+=2*N){
        // 执行异步的数据拷贝
        cudaMemcpyAsync(dev_a0, host_a+i, sizeof(int)*N, cudaMemcpyHostToDevice,stream0);
        cudaMemcpyAsync(dev_a1, host_a+i+N, sizeof(int)*N, cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(dev_b0, host_b+i, sizeof(int)*N, cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(dev_b1, host_b+i+N, sizeof(int)*N, cudaMemcpyHostToDevice, stream1);
        //调用核函数
        kernel<<<grid_size, block_size,0,stream0>>>(dev_a0, dev_b0, dev_c0, N);
        kernel<<<grid_size, block_size,0,stream1>>>(dev_a1, dev_b1, dev_c1, N);
        //流内回传数据
        cudaMemcpyAsync(host_c+i,   dev_c0, sizeof(int)*N, cudaMemcpyDeviceToHost,stream0);
        cudaMemcpyAsync(host_c+i+N, dev_c1, sizeof(int)*N, cudaMemcpyDeviceToHost,stream1);

    }
    // !流内同步在操作定义完成后执行
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    // 记录执行时间
    cudaEventRecord(end);
    //!同步事件
    cudaEventSynchronize(end);
    float elapse = 0.0;
    cudaEventElapsedTime(&elapse, start, end);
    printf("time elapse:%f  ms\n",elapse);
    //销毁流和事件
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    //! 释放所有内存
    // !!!注意这里的内存由于使用cudaHostAlloc分配的锁页内存，要用cudaFreeHost释放；
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a0);
    cudaFree(dev_a1);
    cudaFree(dev_b0);
    cudaFree(dev_b1);
    cudaFree(dev_c0);
    cudaFree(dev_c1);
    return 0;

}