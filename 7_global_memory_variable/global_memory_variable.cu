#include <stdio.h>
#include <stdlib.h>
#include "../tools/common.cuh"

__device__ int d_x = 1;
__device__ int d_y[2];

__global__ void kernel(){
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x, d_y[0], d_y[1] are:%d,%d,%d\n", d_x, d_y[0],d_y[1]);
}

int main(int argc, char * argv[]){
    set_device_by_arg(argc, argv);
    int h_y[2] = {10,20};
    //从主机传数据到device静态内存
    printf("start to trasfer data from host to static device memory\n");
    cudaMemcpyToSymbol(d_y, h_y, sizeof(int)*2);
    //执行核函数
    dim3 block(1);
    dim3 grid(1);
    kernel<<<block, grid>>>();
    //从device传数据到主机
    printf("start ot transfer data from static device memory to host\n");
    cudaDeviceSynchronize();
    cudaMemcpyFromSymbol(h_y, d_y, sizeof(int)*2);
    printf("h_y[0],h_y[1] are:%d,%d\n",h_y[0],h_y[1]);
    printf("done.\n");
    cudaDeviceReset();
    return 0;
}
