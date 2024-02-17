#include <stdio.h>
#include <stdlib.h>
#include "../tools/common.cuh"

#define NUM_REPEATS 10
__device__ float add(const float x, const float y){
    return x+y;
}

__global__ void addFromGPU(float *A, float *B, float *C, const int N){
    const int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid > N) return;
    C[gid] = add(A[gid], B[gid]);
}

void init_data(float *vec, const int num){
    for (int i=0;i<num;i++){
        vec[i] = (float) (rand() & 0xFF)/10.f;
    }
}

int main(void){

    set_device();
    printf("allocate host memory\n");
    int num = 4096;
    size_t size_bytes = num * sizeof(float);

    float *host_A, *host_B, *host_C;
    host_A = (float *)malloc(size_bytes);
    host_B = (float *)malloc(size_bytes);
    host_C = (float *)malloc(size_bytes);

    if (host_A != NULL && host_B != NULL && host_C!= NULL){
        memset(host_A, 0, size_bytes);
        memset(host_B, 0, size_bytes);
        memset(host_C, 0, size_bytes);
    }
    else{
        printf("fail to allocate host memory\n");
        exit(-1);
    }
    printf("allocate device memory\n");
    float *device_A, *device_B, *device_C;
    cuda_error_check(cudaMalloc((float **)&device_A, size_bytes),__FILE__, __LINE__);
    cuda_error_check(cudaMalloc((float **)&device_B, size_bytes),__FILE__, __LINE__);
    cuda_error_check(cudaMalloc((float **)&device_C, size_bytes),__FILE__, __LINE__);
    if (device_A !=NULL && device_B != NULL && device_C != NULL){
        cuda_error_check(cudaMemset(device_A, 0, size_bytes),__FILE__,__LINE__);
        cuda_error_check(cudaMemset(device_B, 0, size_bytes),__FILE__,__LINE__);
        cuda_error_check(cudaMemset(device_C, 0, size_bytes),__FILE__,__LINE__);

    }
    else {
        printf("fail to allocate device memory\n");
        free(host_A);
        free(host_B);
        free(host_C);
        exit(-1);
    }    
    // init data;
    printf("init data\n");
    srand(666);
    init_data(host_A, num);
    init_data(host_B, num);

    // transfer data from host to device
    printf("transfer data\n");
    cuda_error_check(cudaMemcpy(device_A, host_A, size_bytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);
    cuda_error_check(cudaMemcpy(device_B, host_B, size_bytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);
    cuda_error_check(cudaMemcpy(device_C, host_C, size_bytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);

    //call kernel function
    printf("set dim\n");
    dim3 block(32);
    int grid_count = num/32;
    int grid_sur = num%32;
    if (grid_sur != 0){
        grid_count += 1;
    }
    dim3 grid(grid_count);
    
    float t_sum = 0;
    printf("set dim done.\n");

    for (int i=0; i<NUM_REPEATS; i++){

        printf("start call kernel run %d \n",i);
        //首先声明事件；
        cudaEvent_t start, stop;
        //创建事件；
        cuda_error_check(cudaEventCreate(&start),__FILE__, __LINE__);
        cuda_error_check(cudaEventCreate(&stop),__FILE__, __LINE__);
        //创建开始记录；
        cuda_error_check(cudaEventRecord(start),__FILE__,__LINE__);
        //执行核函数
        addFromGPU<<<grid,block>>>(device_A, device_B, device_C, num);
        //执行核函数错误检查
        cuda_error_check(cudaGetLastError(),__FILE__,__LINE__);
        //?不需要执行设备同步吗？似乎同步也没事儿。。。
        cuda_error_check(cudaDeviceSynchronize(),__FILE__,__LINE__);
        //创建结束记录;
        cuda_error_check(cudaEventRecord(stop),__FILE__,__LINE__);
        //同步结束事件；
        cuda_error_check(cudaEventSynchronize(stop), __FILE__, __LINE__);
        float elapsed_time = 0;
        //创建执行时间记录
        cuda_error_check(cudaEventElapsedTime(&elapsed_time, start, stop),__FILE__,__LINE__);
        //记录总时间
        t_sum += elapsed_time;
        //销毁开始、结束事件
        cuda_error_check(cudaEventDestroy(start),__FILE__,__LINE__);
        cuda_error_check(cudaEventDestroy(stop),__FILE__,__LINE__);

    }
    printf("recurrent finished.\n");
    float t_ave = t_sum/NUM_REPEATS;
    printf("elapsed_time:%.2f ms\n",t_ave);
    //transfer data from device to host;
    cuda_error_check(cudaMemcpy(host_C, device_C,size_bytes, cudaMemcpyDeviceToHost),__FILE__, __LINE__);

    //free all memory
    free(host_A);
    free(host_B);
    free(host_C);
    cuda_error_check(cudaFree(device_A),__FILE__, __LINE__);
    cuda_error_check(cudaFree(device_B),__FILE__, __LINE__);
    cuda_error_check(cudaFree(device_C),__FILE__, __LINE__);

    cuda_error_check(cudaDeviceReset(),__FILE__, __LINE__);
    printf("done.\n");
    return 0;
}