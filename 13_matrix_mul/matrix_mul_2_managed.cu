#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>

#define ERROR_CHECK(r){\
    cudaError_t rr = r;\
    if(rr!=cudaSuccess){\
        printf("cuda error:%s, file:%s, line:%s\n",cudaGetErrorString(r), __FILE__, __LINE__);\
        exit(-1);\
    }\
}
// 代码不够稳健，对较小的数组会失败；
// 在非集成显卡上使用统一内存操作，不会省去内存拷贝操作
// 在统一内存上执行uncoalesced memory access,会导致性能下降,一维数组在内存中是连续存储的，可以更好地利用GPU的内存预取和缓存机制。
// 若使用统一内存，实测在二维数组上，global操作时间为155us,在一维数组上global操作时间为48us
// 但如果使用共享内存，则一维或二维数组的性能差异不大，均为3.87us
// 若不使用统一内存，都使用一维数组，则global操作和shared操作时间差异在可接受范围内，global操作时间为4.48us,shared操作时间为3.87us.

#define M 40
#define N 50
#define K 60
#define BLOCK_SIZE 32

__managed__ int A[M][N] = {1};
__managed__ int B[N][K] = {2};
__managed__ int C_g[M][K] = {0};
__managed__ int C_s[M][K] = {0};
__managed__ int C_c[M][K] = {0};

    // 在性能方面，由于一维数组的元素在内存中是连续存储的，因此访问一维数组的元素通常比访问二维数组的元素更高效。
    // 而对于二维数组，由于行与行之间的元素存储不是连续的，可能会引入额外的缓存行失效等开销，导致性能略低于一维数组。
    // 使用一维数组可以更好地适应GPU的内存访问模式。一维数组在内存中是连续存储的，可以更好地利用GPU的内存预取和缓存机制。
    // 在并行计算中，GPU线程可以更有效地访问一维数组的连续内存块，减少内存访问延迟，提高内存带宽利用率。
    // 一维数组的内存访问模式更适合GPU的并行计算模式，可以更好地发挥GPU的计算能力。
__global__ void gpu_matrix_mul_global(int A[M][N],int B[N][K],int C[M][K]){

    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    if(iy < M && ix < K){
        int temp = 0;
        for (int i=0;i<N;i++){
            temp += A[iy][i] * B[i][ix] ;
        }
        C[iy][ix] = temp;
    }
}


__global__ void gpu_matrix_mul_shared(int A[M][N],int B[N][K], int C[M][K]){
    __shared__ int A_temp[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int B_temp[BLOCK_SIZE][BLOCK_SIZE];
    // blockDim.x = BLOCK_SIZE, blockDim.y = BLOCK_SIZE
    int ix = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    int iy = threadIdx.y + blockIdx.y * BLOCK_SIZE;
    int temp = 0;
    // 注意是按照N的大小进行循环
    for(int i=0;i<(N+BLOCK_SIZE-1)/BLOCK_SIZE;i++){
        if (iy < M && i *BLOCK_SIZE + threadIdx.x < N){
            // 取出iy为行坐标点、每个block内对应threadIdx.x对应的A的数
            A_temp[threadIdx.y][threadIdx.x] = A[iy][i *BLOCK_SIZE + threadIdx.x];
        }
        else{
            A_temp[threadIdx.y][threadIdx.x] = 0;
        }
        if (ix < K && i *BLOCK_SIZE + threadIdx.y < N){
            //取出ix为列坐标点,每个block内threadIdx.y对应的B的数
            B_temp[threadIdx.y][threadIdx.x] = B[i*BLOCK_SIZE+threadIdx.y][ix];
        }
        else{
            B_temp[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        for(int k=0;k<BLOCK_SIZE;k++){
            temp += A_temp[threadIdx.y][k] * B_temp[k][threadIdx.x];
        }
        __syncthreads();
    
    }
    //i<gridDim.x循环完毕后，temp即包含了A的iy坐标点所有行，B的ix坐标点所有列分块积的和；
    if (iy < M && ix < K){
        C[iy][ix] = temp;
    }
}


void cpu_matrix_mul(int A[M][N],int B[N][K],int C[M][K]){
    for(int i=0; i<M; i++){
        for (int j=0;j<K;j++){
            int temp = 0;
            for(int k=0;k<N;k++){
                temp += A[i][k] * B[k][j];
            }
            C[i][j] = temp;
        }
    }
}


double get_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    double _time = (double)(tv.tv_sec+tv.tv_usec*0.000001);
    return _time;
}

int main(){
    //! 注意N和M的位置
    dim3 grid_size((N+BLOCK_SIZE-1)/BLOCK_SIZE,(M+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 block_size(BLOCK_SIZE,BLOCK_SIZE);
    //init data,pass;
    //执行kernel函数
    double global_start = get_time();
    gpu_matrix_mul_global<<<grid_size,block_size>>>(A,B,C_g);
    double global_end = get_time();

    double shared_start = get_time();
    gpu_matrix_mul_shared<<<grid_size,block_size>>>(A,B,C_s);
    double shared_end = get_time();

    double cpu_start = get_time();
    cpu_matrix_mul(A,B,C_c);
    double cpu_end = get_time();

    for(int i=0;i<M;i++){
        for(int j=0;j<K;j++){
            if(
                (C_g[i][j] != C_s[i][j]) |\
                (C_g[i][j] != C_c[i][j]) | \
                (C_s[i][j] != C_c[i][j])
             ){
                printf("test failed\n");
                exit(-1);
             }

        }
    }
    printf("test pass\n");
    double global_elap = global_end - global_start;
    double shared_elap = shared_end - shared_start;
    double cpu_elap = cpu_end - cpu_start;
    // gpu global time elapse:0.063056ms,gpu shared time elapse:0.000001ms, cpu time elapse:0.001074ms
    // gpu global :58.707214,gpu shared :0.000888
    printf("gpu global time elapse:%fms,gpu shared time elapse:%fms, cpu time elapse:%fms\n",global_elap, shared_elap, cpu_elap);
    printf("gpu global :%f,gpu shared :%f\n",global_elap/cpu_elap, shared_elap/cpu_elap);
    return 0;
    
}