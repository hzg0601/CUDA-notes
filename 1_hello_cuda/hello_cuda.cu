#include <stdio.h>

__global__ void hello_cuda2(){
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid*blockDim.x+tid;
    printf("hello cuda2 id: %d, tid: %d, bid:%d\n", idx,tid,bid);

}
int main(){
    printf("hello cup\n");
    hello_cuda<<<2,32>>>();
    hello_cuda2<<<2,32>>>();
    cudaDeviceSynchronize();
    return 0;
}
