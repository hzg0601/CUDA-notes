#include <stdio.h>
#include <stdlib.h>
#include <time.h>   


#define BLOCK_SIZE 256
#define N 1000000
#define GRID_SIZE  ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) 
#define topk 20


__managed__ int source_array[N];
__managed__ int _1pass_results[topk * GRID_SIZE];
__managed__ int final_results[topk];

__device__ __host__ void insert_value(int* array, int k, int data)
{
    for (int i = 0; i < k; i++)
    {
        if (array[i] == data)
        {
            return;
        }
    }
    if (data < array[k - 1])
        return;
    for (int i = k - 2; i >= 0; i--)
    {
        if (data > array[i])
            array[i + 1] = array[i];
        else {
            array[i + 1] = data;
            return;
        }
    }
    array[0] = data;
}

__global__ void top_k(int* input, int length, int* output)
{
    //need your cuda code here
    __shared__ int temp_per_block[BLOCK_SIZE * topk];

    //printf("%d\n",k);
    int idx = threadIdx.x + blockIdx.x *blockDim.x ;
    //先初始化，并且处理单个个线程的TOPN
    int t[topk] = {0};   
    for(int j = idx; j < length ; j +=  gridDim.x *blockDim.x)
    {

        if(input[j] > t[topk-1]){
        //printf("%d",input[j]);
            t[topk-1] = input[j];
            for(int i = topk-2 ; i >= 0 ; i --)
            {
                if( t[i + 1] > t[i] )
                {
                    int temp = t[i+1];
                    t[i+1] = t[i];
                    t[i] = temp;  
                } 
            }
        }
    }
    //printf("%d\n",t[1]);
    //thread 存入block
    for(int i = 0; i < topk ; i ++)
    {
        temp_per_block[threadIdx.x *topk+i ] = t[i];
        //printf("%d\n",t[i]);
    }
    __syncthreads();
    //printf("!!");

    //处理整个block的
    for( int i = BLOCK_SIZE / 2  ; i >= 1  ; i /= 2){
        if( threadIdx.x < i )
        {
            //a 是第二个线程所存数据的开头 
            int a = (threadIdx.x+i)*topk;
            // 循环n次，将a开始的n个数字都和b所在的数字进行大小对比
            for(int j = 0;  j < topk ; j ++){
                //如果 a 位置数字更大 进行替换
                if(temp_per_block[ a + j  ] > t[ topk - 1]){
                    //执行替换，替换完成后循环n次插入对应位置
                    t[topk-1] = temp_per_block[ a + j  ];
                    for(int i = topk-2 ; i >= 0 ; i --)
                    {
                        if( t[i + 1] > t[i] )
                        {
                            int temp = t[i+1];
                            t[i+1] = t[i];
                            t[i] = temp;  
                        } 
                    }
                } 
            }
        }
        __syncthreads();
        if(threadIdx.x < i){
            for(int j = 0; j < topk ; j ++)
            {
                temp_per_block[threadIdx.x*topk  + j ] = t[j];
            }
        }
        __syncthreads();
    }  
    __syncthreads();
    if( threadIdx.x==0&&blockDim.x * blockIdx.x < length)
    {
        for(int i = 0; i < topk ; i ++)
        {
            output[blockIdx.x*topk + i] = t[i];
        }
        __syncthreads();
    }
}

void cpu_result_topk(int* input, int count, int* output)
{
    for (int i = 0; i < count; i++)
    {
        insert_value(output, topk, input[i]);

    }
}

void _init(int* ptr, int count)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < count; i++) ptr[i] = rand();
}

int main(int argc, char const* argv[])
{
    int cpu_result[topk] = { 0 };
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Fill input data buffer
    _init(source_array, N);


    printf("\n***********GPU RUN**************\n");
    cudaEventRecord(start);
    top_k << <GRID_SIZE, BLOCK_SIZE >> > (source_array, N, _1pass_results);
    cudaGetLastError();
    top_k << <1, BLOCK_SIZE >> > (_1pass_results, topk * GRID_SIZE, final_results);
    cudaGetLastError();
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Time = %g ms.\n", elapsed_time);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cpu_result_topk(source_array, N, cpu_result);


    int ok = 1;
    for (int i = 0; i < topk; ++i)
    {
        printf("cpu top%d: %d; gpu top%d: %d \n", i + 1, cpu_result[i], i + 1, final_results[i]);
        if (fabs(cpu_result[i] - final_results[i]) > (1.0e-10))
        {

            ok = 0;
        }
    }

    if (ok)
    {
        printf("Pass!!!\n");
    }
    else
    {
        printf("Error!!!\n");
    }
    return 0;
}