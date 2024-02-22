#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <iostream>
using namespace std;

#define ERROR_CHECK(r)\
{\
    cudaError_t rr = r;\
    if (rr != cudaSuccess) {\
    fprintf(stderr,"CUDA ERROR %s, file:%s, functon:%s, line:%s\n",cudaGetErrorString(rr),__FILE__,__FUNCTION__,__LINE__);\
    exit(-1);\
    }\
}

#define BLOCK_SIZE 256
#define N 100000
#define GRID_SIZE ((N+BLOCK_SIZE-1)/BLOCK_SIZE)
#define topk 20

//按倒序插入数据
__device__ __host__ void insert_value(int * array, const int k, const int data){
    //如下代码保证无重复插入
    // for(int i=0; i<k; i++){
    //     if (array[i] == data){
    //         return;
    //     }
    // }
    if (data < array[k-1]) return;

    for (int i=k-2; i>=0; i--){
        if (data > array[i]){
            array[i+1] = array[i];
        }
        else{
            array[i+1] = data;
            return;
        }
    }
    array[0] = data;
}
// 先对gid相同的哪些数进行排序，只排序topk个
__global__ void gpu_get_topk(int * array, const int num, int * output){
    
    __shared__ int temp_per_block[BLOCK_SIZE * topk];
    int glid = threadIdx.x + blockIdx.x * blockDim.x;
    // 先对gid相同的那些数进行排序，只对单个线程负责的数进行排序，只取topk个，
    int glid_rank[topk] = {0};
    for(int j = glid; j<num;j+= gridDim.x * blockDim.x){
        if(array[j] > glid_rank[topk-1]){
            glid_rank[topk-1] = array[j];
            for(int i=topk-2;i>=0;i--){
                if (glid_rank[i]< glid_rank[i+1]){
                    int temp = glid_rank[i+1];
                    glid_rank[i+1] = glid_rank[i];
                    glid_rank[i] = temp;
                }
            }
        }
    }
    //每个线程得到一个topk的数组；

    //将一个block内所有thread的排序结果存入共享内存，按线程的id进行存放；
    for(int i=0;i<topk;i++){
        temp_per_block[threadIdx.x *topk + i] = glid_rank[i];
    }
    __syncthreads();
    // |____|____|____|____| 
    //       <-------|
    //           <----------|   
    // |____| |____| |____| |____|
    // 每个线程独有的t,根据id的排序，与temp_per_block一致,
    // 因此只需进行折半比较，即用temp_per_block的后半段与block内的前半部分线程比较，即完成了一轮比较
    // 而由于t是已排序好的，因此只需要比较temp_per_block的顺序和对应t的最后一个即可，比较后重排t即完成插入
    for(int i = BLOCK_SIZE/2; i>=1; i/=2){
        if(threadIdx.x < i){
            int offset = (threadIdx.x + i) * topk;
            for(int j=0; j<topk;j++){
                if (temp_per_block[offset+j] > glid_rank[topk-1]){
                    glid_rank[topk-1] = temp_per_block[offset+j];
                    for (int i=topk-2;i>=0;i--){
                        if(glid_rank[i+1]>glid_rank[i]){
                            int temp = glid_rank[i+1];
                            glid_rank[i+1] = glid_rank[i];
                            glid_rank[i] = temp;
                        }
                    }
                }
            }
        }
        __syncthreads();
        //完成一轮排序后，将新排序的glid_rank赋值给temp_per_block
        if(threadIdx.x < i){
            for(int j=0; j<topk;j++){
                temp_per_block[threadIdx.x*topk + j] = glid_rank[j];
            }
        }
        __syncthreads();
        //上述步骤完成后，每个block即完全排序好的，然后取出每个block的topk存入output
    }
    __syncthreads();
    if(threadIdx.x==0 && blockDim.x * blockIdx.x < num){
        for (int i=0;i<topk;i++){
        output[blockIdx.x * topk + i] = glid_rank[i];
        }
        __syncthreads();
    }
}

void cpu_get_topk(int * array, int k, int * result){
    for(int i=0; i< k; i++){
        insert_value(result, topk, array[i]);
    }
}

void init_data(int * array, int k){
    // srand((unsigned)time(NULL));
    for (int i=0; i<k;i++) array[i] = rand()%100000;
}

double get_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return ((double) tv.tv_usec*0.000001 + tv.tv_sec);
}

int main(){
    //设置GPU，可选；
    //初始化host数据；
    printf("start to initialize host memory..\n");
    size_t size_bytes_array = N * sizeof(int);
    
    int * host_array, *host_final_result;
    host_array = (int *)malloc(size_bytes_array);
    host_final_result = (int *)malloc(topk * sizeof(int));
    init_data(host_array, N);
    memset(host_final_result,0,topk*sizeof(int));
    //初始化device数据；
    printf("start to initialize device memory..\n");
    size_t size_bytes_result = topk * GRID_SIZE * sizeof(int);
    int * device_array, * device_result, *device_final_result;
    cudaMalloc((int **)&device_array, size_bytes_array);
    cudaMalloc((int **)&device_result, size_bytes_result);
    cudaMalloc((int **)&device_final_result,topk*sizeof(int));
    cudaMemset(device_array, 0, size_bytes_array);
    cudaMemset(device_result, 0, size_bytes_result);
    cudaMemset(device_final_result,0,topk*sizeof(int));

    int cpu_final_result[topk] ={0};

    //转移数据
    printf("start to memory copy..\n");
    try{
    cudaMemcpy(device_array, host_array, size_bytes_array, cudaMemcpyHostToDevice);
    //调用kernel函数；
    printf("start to call kernel functon..\n");
    double gpu_start = get_time();
    gpu_get_topk<<<GRID_SIZE,BLOCK_SIZE>>>(device_array, N, device_result);
    ERROR_CHECK(cudaGetLastError());
    gpu_get_topk<<<1, BLOCK_SIZE>>>(device_result, GRID_SIZE*topk,device_final_result);
    ERROR_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();
    // 取回数据
    printf("start to write back data..\n");
    cudaMemcpy(host_final_result,device_final_result,topk*sizeof(int),cudaMemcpyDeviceToHost);
    // for (int i=0;i<topk;i++){
    //     cout <<"host final result:"<<i<<":"<<host_final_result[i]<<endl;
    //     cout <<"cpu final result:"<<i<<":"<<cpu_final_result[i]<<endl;
    // }
    
    // cudaMemcpy(device_result, host_result, size_bytes_result, cudaMemcpyDeviceToDevice);
    // cpu_get_topk(host_result, BLOCK_SIZE*topk, final_result);

    double gpu_end = get_time();
    double gpu_elapse = gpu_end - gpu_start;
    printf("start to execute code on cpu..\n");

    double cpu_start = get_time();
    cpu_get_topk(host_array, N, cpu_final_result);
    double cpu_end = get_time();
    double cpu_elapse = cpu_end - cpu_start;
    // cout<<"cpu_elapse:"<<cpu_elapse<<endl;
    // cout<<"gpu_elapse:"<<gpu_elapse<<endl;
    printf("gpu_elapse:%g,cpu_elapse:%g s\n",gpu_elapse,cpu_elapse);
    for(int i=0; i<topk; i++){
        printf("cpu result[%d]:%d,gpu result[%d]:%d\n",i,cpu_final_result[i],i,host_final_result[i]);
        }
    }
    //回收全部内存；
    catch(...){
        printf("ERROR OCCUR\n");
        free(host_array);
        free(host_final_result);

        cudaFree(device_array);
        cudaFree(device_result);
        cudaFree(device_final_result);
        cudaDeviceReset();
        exit(-1);
    }
    return 0;
}