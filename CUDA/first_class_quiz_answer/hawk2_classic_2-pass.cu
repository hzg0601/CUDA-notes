#include<stdio.h>
#include<stdint.h>
#include<time.h>     //for time()
#include<stdlib.h>   //for srand()/rand()
#include<sys/time.h> //for gettimeofday()/struct timeval


#define KEN_CHECK(r) \
{\
    cudaError_t rr = r;   \
    if (rr != cudaSuccess)\
    {\
        fprintf(stderr, "CUDA Error %s, function: %s, line: %d\n",       \
		        cudaGetErrorString(rr), __FUNCTION__, __LINE__); \
        exit(-1);\
    }\
}

#define N 10000000
#define BLOCK_SIZE 256
#define BLOCKS ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) //try next line if you can
//#define BLOCKS 666

__managed__ int source[N];                   //input data
__managed__ int _partial_results[2 * BLOCKS];//for 2-pass kernel
__managed__ int final_result[2 * 1];         //scalar output


__global__ void _hawk_minmax_gpu(int *input, int count,
	                         int *output)
{
    __shared__ int leo_min[BLOCK_SIZE];
    __shared__ int leo_max[BLOCK_SIZE];

    //**********register min/max stage************
    int zpei_min = INT_MAX; 
    int zpei_max = INT_MIN; //caution! 
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
         idx < count;
	 idx += gridDim.x * blockDim.x
	)
    {
	zpei_min = min(zpei_min, input[idx]);
	zpei_max = max(zpei_max, input[idx]);
    }
    //saving the per-thread min/max values to shared memory: leo_min/max[]
    leo_min[threadIdx.x] = zpei_min;
    leo_max[threadIdx.x] = zpei_max;
    __syncthreads();

    //**********shared memory min/max stage***********
    for (int length = BLOCK_SIZE / 2; length >= 1; length /= 2)
    {
        int CJW_min;
	int CJW_max;
	if (threadIdx.x < length)
	{
	    CJW_min = min(leo_min[threadIdx.x], leo_min[threadIdx.x + length]);
	    CJW_max = max(leo_max[threadIdx.x], leo_max[threadIdx.x + length]);
	}
	__syncthreads();  //why we need two __syncthreads() here, and,
	
	if (threadIdx.x < length)
	{
	    leo_min[threadIdx.x] = CJW_min;
	    leo_max[threadIdx.x] = CJW_max;
	}
	__syncthreads();  //....here ?
	
    } //the per-block partial min/max is leo_min[0] & leo_max[0]

    if (blockDim.x * blockIdx.x < count) //in case that our users are naughty
    {
        //per-block results written back, by thread 0, on behalf of a block.
        if (threadIdx.x == 0)
	{
	    output[2 * blockIdx.x + 0] = leo_min[0]; //tell me why
	    output[2 * blockIdx.x + 1] = leo_max[0];
	}
    }
}

typedef struct
{
    int min;
    int max;
}cpu_result_t;	
	
cpu_result_t _hawk_minmax_cpu(int *ptr, int count)
{
    int YZP_min = INT_MAX;
    int YZP_max = INT_MIN;
    for (int i = 0; i < count; i++)
    {
	YZP_min = min(YZP_min, ptr[i]);
	YZP_max = max(YZP_max, ptr[i]);
    }

    cpu_result_t r;
    {
	r.min = YZP_min;
	r.max = YZP_max;
    }		
    return r;
}

void _nanana_init(int *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
    //for (int i = 0; i < count; i++) ptr[i] = (rand() << 3) ^ rand();
    for (int i = 0; i < count; i++) ptr[i] = rand();
}

double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return ((double)tv.tv_usec * 0.000001 + tv.tv_sec);
}

int main()
{
    //**********************************
    fprintf(stderr, "nanana is filling the buffer with %d elements...\n", N);
    _nanana_init(source, N);

    //**********************************
    //Now we are going to kick start your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!
    //Good luck & have fun!
    
    fprintf(stderr, "Running on GPU...\n");
    
double t0 = get_time();
    _hawk_minmax_gpu<<<BLOCKS, BLOCK_SIZE>>>(source, N, _partial_results);
        KEN_CHECK(cudaGetLastError());  //checking for launch failures
	
    _hawk_minmax_gpu<<<1, BLOCK_SIZE>>>(_partial_results, 2 * BLOCKS,
     		        		final_result);
        KEN_CHECK(cudaGetLastError());  //the same
	
    KEN_CHECK(cudaDeviceSynchronize()); //checking for run-time failurs
double t1 = get_time();

    int A0 = final_result[0];
    int A1 = final_result[1];
    fprintf(stderr, "GPU min: %d, max: %d\n", A0, A1);


    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

double t2 = get_time();
    cpu_result_t B = _hawk_minmax_cpu(source, N);
double t3 = get_time();
    fprintf(stderr, "CPU min: %d, max: %d\n", B.min, B.max);

    //******The last judgement**********
    if (A0 == B.min && A1 == B.max)
    {
        fprintf(stderr, "Test Passed!\n");
    }
    else
    {
        fprintf(stderr, "Test failed!\n");
	exit(-1);
    }
    
    //****and some timing details*******
    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);

    return 0;
}	
	
