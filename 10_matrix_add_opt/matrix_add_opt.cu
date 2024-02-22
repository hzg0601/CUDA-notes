#include <stdio.h>
#include <stdint.h>
#include <stdlib.h> //for srand, rand
#include <time.h>
#include <sys/time.h> // for gettimeofday
// 每次调用kernel函数后使用cudaGetLastError，调用其他设备函数直接包裹
#define ERROR_CHECK(r) \
{\
    cudaError_t rr = r; \
    if (rr != cudaSuccess) \
    {\
        fprintf(stderr, "CUDA ERROR %s, file:%s, function: %s, line:%d\n", \
                cudaGetErrorString(rr),__FILE__, __FUNCTION__, __LINE__);\
                exit(-1);\
    }\
}

#define N 1000000
#define BLOCK_SIZE 256
#define NUM_BLOCKS ((N+BLOCK_SIZE-1) /BLOCK_SIZE)

// 先执行相同gid的求和，再执行block_wise加法，注意保证在n<gid时，不对大于n的那些元素进行操作；
__global__ void _sum_gpu(int *A, const int n, int * result){
    // 参数首先被放在寄存器上，寄存器如果放不下就会放在本地内存上，而本地内存是全局内存的一部分；
    // 共享内存是block内共享的
    //每个核函数执行时，CUDA运行时系统会为每个线程分配相应的threadIdx、blockIdx和gridDim值，
    //这些值在核函数的执行期间保持不变。当执行完一个核函数后，进入下一个核函数时，这些变量的值会重新设置。
    __shared__ int threadwise_sum[BLOCK_SIZE];
    // register summation, 计算所有block中线程id相同的元素的和
    int partial_sum = 0;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int glid = tid + bid * blockDim.x; //计算其全局id
    for (int gid = glid;gid < n; gid += gridDim.x*blockDim.x){ 
        //如果全局id<n,则令全局id+block数目*所在block_id,即若元素个数大于总线程数，则令具有每个线程做多个计算
        //从而每个partitial_sum都是相同gid的数的和；
        partial_sum += A[gid];
    }
    //由于每个__shared__变量是block可见的，因此只需申请BLOCK_SIZE大小的共享内存
    //相同tid的gid的数的和按block存储到共享内存中，每个核函数定义的共享内存变量在block内是独立的；
    //故而仍是定义了grid.dim 个共享变量，每个共享变量存储block_size个数字
    threadwise_sum[tid] = partial_sum;
    __syncthreads();
    // shared memory summation, 对每个block执行间域加法，避免线程束分化
    for (int length = BLOCK_SIZE/2; length >=1; length/=2){
        int double_half = -1;
        if (tid < length){
            double_half = threadwise_sum[tid] + threadwise_sum[tid + length];
        }
        __syncthreads();
        if (tid < length){
            //每次对share memory变量执行一个操作都要进行线程同步
            threadwise_sum[tid] = double_half;
        }
        __syncthreads();  
    }

    // 将每个block的和赋值给result[bid],tid == 0 确保只执行一次赋值
    //blockDim.x * blockIdx.x <n，确保赋值发生在[0，gid]之内，即在n<gid时，不对大于n的那些元素进行操作；
    if (blockDim.x * blockIdx.x < n){
        if (tid == 0) result[bid] = threadwise_sum[0];
    }
}


int _sum_cpu(int * ptr, const int n){
    int sum = 0;
    for (int i = 0; i< n; i++){
        sum += ptr[i];
    }
    return sum;
}
    
void init_data(int * ptr, const int n){
    srand(666);
    for (int i = 0; i<n; i++) ptr[i] = rand();
}

double get_time(){
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001+tv.tv_sec); 
}

int main(int argc, char *argv[]){
    //定义使用的gpu,可选；
    int device_count = 0;
    int devcie_id = 0;
    cudaGetDeviceCount(& device_count);
    if (argc > 1 ) {
        if (device_count > atoi(argv[1])) devcie_id = atoi(argv[1]);
        cudaSetDevice(devcie_id);
    }
    //定义Host变量，并初始化;
    int size_bytes = N * sizeof(int);
    int * host_array, *host_result;
    host_array = (int *)malloc(size_bytes);
    host_result = (int *)malloc(NUM_BLOCKS * sizeof(int));
    init_data(host_array, N);
    memset(host_result, 0, NUM_BLOCKS*sizeof(int));
    //host变量转移至设备；
    int * device_array, *device_result;
    cudaMalloc((int **) & device_array,size_bytes);
    cudaMalloc((int **)& device_result,NUM_BLOCKS * sizeof(int));
    cudaMemset(device_array, 0, size_bytes);
    cudaMemset(device_result,0, BLOCK_SIZE * sizeof(int));

    cudaMemcpy(device_array, host_array, size_bytes, cudaMemcpyHostToDevice);
    //调用kernel函数；
    double gpu_start_time = get_time();
    _sum_gpu<<<NUM_BLOCKS,BLOCK_SIZE>>>(device_array,N, device_result);
    //同步设备；
    // ERROR_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    //设备数据转移至host;
    cudaMemcpy(host_result, device_result, NUM_BLOCKS*sizeof(int), cudaMemcpyDeviceToHost);
    int gpu_sum = _sum_cpu(host_result, NUM_BLOCKS);
    double gpu_end_time = get_time();
    double gpu_elapse = gpu_end_time - gpu_start_time;

    double cpu_start_time = get_time();
    int cpu_sum = _sum_cpu(host_array, N);
    double cpu_end_time = get_time();
    double cpu_elapse = cpu_end_time - cpu_start_time;
    //打印结果；
    printf("gpu result:%d, gpu elapse:%fms\n",gpu_sum, gpu_elapse);
    printf("cpu result:%d, cpu elapse:%fms\n",cpu_sum, cpu_elapse);
    //回收设备、host内存；
    free(host_array);
    free(host_result);
    cudaFree(device_array);
    cudaFree(device_result);
    //重置cudaDeviceReset;
    cudaDeviceReset();
    return 0;
}