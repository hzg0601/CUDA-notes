#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>

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
    for(int i=0; i<k; i++){
        if (array[i] == data) return;
    }
    if (data < array[k-1]) return;

    for (int i=(k-2); i>=0; i--){
        if (data > array[i]){
            array[i+1] = array[i];
        }
        else{
            array[i+1] = data;
            return;
        }
    }
}
// 先对gid相同的哪些数进行排序，只排序topk个
__global__ void gpu_get_topk(int * array, const int num, int * output){
    
    __shared__ int temp_per_block[BLOCK_SIZE * topk];
    int glid = threadIdx.x + blockIdx.x * blockDim.x;
    // 先对gid相同的那些数进行排序，只对单个线程负责的数进行排序，只取topk个，
    int glid_rank[topk] = {0};
    for(int i = glid; i<num;i+= gridDim.x * blockDim.x){
        if(array[i] > glid_rank[topk-1]){
            glid_rank[topk-1] = array[i];
            for(int j=topk-2;j>=0;j++){
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
    for(int i = BLOCK_SIZE/2;i>=1;i/2){
        if(threadIdx.x < i){
            int offset = (threadIdx.x + i) * topk;
            for(int j=0; j<topk;j++){
                if (temp_per_block[offset+j] > glid_rank[topk-1]){
                    glid_rank[topk-1] = temp_per_block[offset+j];
                    for (int k=topk-2;k>=0;k--){
                        if(glid_rank[+1]>glid_rank[k]){
                            int temp = glid_rank[k+1];
                            glid_rank[k+1] = glid_rank[k];
                            glid_rank[k] = temp;
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
    if(threadIdx.x==0 && blockDim.x * gridDim.x < num){
        for (int j=0;j<topk;j++){
        output[blockDim.x * topk + j] = temp_per_block[j];
        }
    }

}

void cpu_get_topk(int * array, int k, int * result){
    for(int i=0; i< k; i++){
        insert_value(result, topk, array[i]);
    }
}

void init_data(int * array, int k){
    srand((unsigned)time(NULL));
    for (int i=0; i<k;i++) array[i] = rand();
}

double get_time(){
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return ((double) tv.tv_usec*0.000001 + tv.tv_sec);
}

int main(){
    //设置GPU，可选；
    //初始化host数据；
    size_t size_bytes_array = N * sizeof(int);
    size_t size_bytes_result = topk * GRID_SIZE * sizeof(int);
    int * host_array, * host_result, *final_result;
    host_array = (int *)malloc(size_bytes_array);
    host_result = (int *) malloc( size_bytes_result);
    final_result = (int *)malloc(topk * sizeof(int));
    init_data(host_array, N);
    memset(host_result, 0, size_bytes_result);
    //初始化device数据；

    int * device_array, * device_result;
    cudaMalloc((int **)&device_array, size_bytes_array);
    cudaMalloc((int **)&device_result, size_bytes_result);
    cudaMemset(device_array, 0, size_bytes_array);
    cudaMemset(device_result, 0, size_bytes_result);
    //转移数据

    cudaMemcpy(device_array, host_array, size_bytes_array, cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, host_result, size_bytes_result, cudaMemcpyHostToDevice);
    //调用kernel函数；
    double gpu_start = get_time();
    gpu_get_topk<<<GRID_SIZE,BLOCK_SIZE>>>(device_array, N, device_result);
    cudaDeviceSynchronize();
    // 取回数据
    cudaMemcpy(device_result, host_result, size_bytes_result, cudaMemcpyDeviceToDevice);

    cpu_get_topk(host_result, BLOCK_SIZE*topk, final_result);

    double gpu_end = get_time();
    double gpu_elapse = gpu_end - gpu_start;
    int * cpu_final_result;
    cpu_final_result = (int *)malloc(topk * sizeof(int));
    double cpu_start = get_time();
    cpu_get_topk(host_array, N, cpu_final_result);
    double cpu_end = get_time();
    double cpu_elapse = cpu_end - cpu_start;
    for(int i=0; i<topk; i++){
    printf("cpu result[i]:%d,cpu_elapse:%s s\n",cpu_final_result[i], cpu_elapse);
    printf("gpu result[i]:%d,gpu_elapse:%s s\n",final_result[i], gpu_elapse);
    }
    //回收全部内存；
    free(host_array);
    free(host_result);
    free(cpu_final_result);
    cudaFree(device_array);
    cudaFree(device_result);
    cudaDeviceReset();
    return 0;
}