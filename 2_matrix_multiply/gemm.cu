#include <stdio.h>
#include "../tools/common.cuh"


__global__ void print_id() {
    printf("in kernel currently\n");
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = tid + bid * blockDim.x;
    printf("The global id, thread id, block id is: %d,%d,%d, respectively\n", gid,tid,bid );
}

__global__ void add_from_gpu(float *A, float *B, float *C, const int N){
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int gid = bid * blockDim.x + tid;
    C[gid] = A[gid] + B[gid];

}

void init_data(float *addr, int elemet_count){
    for (int i = 0; i<elemet_count;i++){
        addr[i] = (float)(rand() & 0xFF)/10.f;
    }
}

void add_from_cpu(float *A, float *B, float *C, const int N){
    for (int i = 0;i < N; i++) C[i] = A[i] + B[i];
}


int main(void){
    // 1. 设置GPU
    set_device();

    // 2. 分配内存
    printf("start to allocate host memory\n");
    int elemet_count = 512;
    size_t size_bytes = elemet_count * sizeof(float);
    float *host_A, *host_B, *host_C;
    host_A = (float *)malloc(size_bytes);
    host_B = (float *)malloc(size_bytes);
    host_C = (float *)malloc(size_bytes);
    //2.1 初始化主机内存
    if (host_A != NULL && host_B !=NULL && host_C != NULL){
        memset(host_A, 0, size_bytes);
        memset(host_B, 0, size_bytes);
        memset(host_C, 0, size_bytes);
    }
    else{
        printf("Fail to initialize host memory\n");
        exit(-1);
    }

    // 3. 分配设备内存
    printf("start to allocate device memory\n");
    float *device_A, *device_B, *device_C;
    cudaMalloc((float **)&device_A, size_bytes);
    cudaMalloc((float **)&device_B, size_bytes);
    cudaMalloc((float **)&device_C, size_bytes);
    if (device_A != NULL && device_B != NULL && device_C !=NULL){
        cudaMemset(device_A, 0, size_bytes);
        cudaMemset(device_B, 0, size_bytes);
        cudaMemset(device_C, 0, size_bytes);

    }
    else{
        printf("Initialize device memory failed!");
        free(host_A);
        free(host_B);
        free(host_C);
        exit(-1);
    }
    // 初始化主机中的数组
    printf("start to initialize host data\n");
    srand(666);
    init_data(host_A, elemet_count);
    init_data(host_B, elemet_count);

    // 4. 数据从主机复制到设备；
    printf("start to memory copy\n");
    cudaMemcpy(device_A, host_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, size_bytes, cudaMemcpyHostToDevice);

    // 4. 调用核函数进行计算

    // 4.1 定义gridDim, blockDim;
    printf("start to call kernel\n");
    dim3 block(32);
    int grid_count = elemet_count/32;
    int grid_sur = elemet_count%32;
    if (grid_sur != 0){
        grid_count += 1;
    }
    dim3 grid(grid_count);

    // 5. 调用核函数；
    print_id<<<2,32>>>();
    add_from_gpu<<<grid, block>>>(device_A, device_B, device_C, elemet_count);
    cudaDeviceSynchronize();
    //6. 计算结果传回主机

    printf("start to transfer back to host\n");
    cudaMemcpy(host_C, device_C, size_bytes, cudaMemcpyDeviceToHost);

    // 打印结果

    for (int i=0;i<10;i++){
        printf("idx=%2d \t maxtrix_A:%-6.2f\t matrix_B:%-6.2f\t matrix_C:%-6.2f\t\n",i+1,host_A[i],host_B[i],host_C[i]);
    }
    printf("execute from cpu \n");
    add_from_cpu(host_A, host_B, host_C, elemet_count);

    for (int i=0;i<10;i++){
        printf("idx=%2d \t maxtrix_A:%-6.2f\t matrix_B:%-6.2f\t matrix_C:%-6.2f\t\n",i+1,host_A[i],host_B[i],host_C[i]);
    }
    printf("free memory\n");
    //7. 释放主机与设备内存
    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    free(host_A);
    free(host_B);
    free(host_C);
    printf("执行完毕\n");

    return 0;

}