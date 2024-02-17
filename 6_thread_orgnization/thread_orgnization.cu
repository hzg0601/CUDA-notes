#include <stdio.h>
#include <stdlib.h>
#include "../tools/common.cuh"

__global__ void add_matrix(int *A, int *B, int *C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.y;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int gid = nx * iy + ix;
    if (ix < nx && iy < ny){
        C[gid] = A[gid] + B[gid];
    }
}

void init_data(int *matrix, const int num){
    srand(666);
    for (int i=0;i<num;i++){
        matrix[i] = (int) (rand() & 0xFF)/10.f;
    }
}

int main(int argc, char * argv[]){
    // set device
    int device_id = 0;
    if (argc > 1) device_id = atoi(argv[1]);
    set_device(device_id);

    // set size
    const int nx = 35;
    const int ny = 17;
    const int nxy = nx * ny;
    size_t size_bytes = nxy * sizeof(int);

    //initialize host memory
    int *host_A, *host_B, *host_C;
    host_A = (int *)malloc(size_bytes);
    host_B = (int *)malloc(size_bytes);
    host_C = (int *)malloc(size_bytes);
    if (host_A != NULL && host_B != NULL && host_C!=NULL){
        memset(host_A, 0, size_bytes);
        memset(host_B, 0, size_bytes);
        memset(host_C, 0, size_bytes);
    }
    else printf("fail to allocate host memory\n");

    // initialize device memory
    int *device_A, *device_B, *device_C;
    cudaMalloc((int **)&device_A, size_bytes);
    cudaMalloc((int **)&device_B,size_bytes);
    cudaMalloc((int **)&device_C, size_bytes);
    if (device_A != NULL && device_B != NULL && device_C != NULL){
        cudaMemset(device_A, 0, size_bytes);
        cudaMemset(device_B, 0, size_bytes);
        cudaMemset(device_C, 0, size_bytes);
    }
    else printf("fail to allocate device memory\n");

    // init host data
    init_data(host_A,nxy);
    init_data(host_B,nxy);

    // transfer data from host to device;
    cudaMemcpy(device_A, host_A, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_B, host_B, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_C, host_C, size_bytes, cudaMemcpyHostToDevice);

    // call kernel function

    dim3 block(8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y);
    add_matrix<<<block, grid>>>(device_A, device_B, device_C, nx, ny);
    cudaDeviceSynchronize();

    cudaMemcpy(host_C, device_C, size_bytes, cudaMemcpyDeviceToHost);

    for (int i=0;i<nxy;i++){
        printf("id=%d,matrix_A=%d,matrix_B=%d,matrix_C=%d\n", i,host_A[i],host_B[i],host_C[i]);
    }
    //free 
    free(host_A);
    free(host_B);
    free(host_C);

    cudaFree(device_A);
    cudaFree(device_B);
    cudaFree(device_C);
    cudaDeviceReset();
    printf("done.\n");
    return 0;

}