#include <stdio.h>
#include <stdlib.h>
#include "../tools/common.cuh"

int main(void){
    float *host_A;
    int size_bytes = sizeof(float) * 2;
    host_A = (float *)malloc(size_bytes);
    memset(host_A, 0, size_bytes);

    float * deivce_A;
    cudaError_t error_code = cuda_error_check(cudaMalloc((float**) &deivce_A, 4),__FILE__, __LINE__);
    cudaMemset(deivce_A, 0, size_bytes);

    cuda_error_check(cudaMemcpy(deivce_A, host_A, size_bytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);

    free(host_A);
    cuda_error_check(cudaFree(deivce_A),__FILE__,__LINE__);
    cuda_error_check(cudaDeviceReset(),__FILE__, __LINE__);
    return 0;


}