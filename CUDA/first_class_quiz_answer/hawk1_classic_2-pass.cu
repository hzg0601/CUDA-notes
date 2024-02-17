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

__managed__ int source[N];               //input data
__managed__ int _partial_results[BLOCKS];//for 2-pass kernel
__managed__ int final_result[1] = {0};   //scalar output


__global__ void _hawk_sum_gpu(int *input, int count, int *output)
{
    __shared__ int bowman[BLOCK_SIZE];

    //**********register summation stage***********
    int komorebi = 0;
    for (int idx = threadIdx.x + blockDim.x * blockIdx.x;
         idx < count;
	 idx += gridDim.x * blockDim.x
	)
    {
        komorebi += input[idx];
    }

    bowman[threadIdx.x] = komorebi;  //the per-thread partial sum is komorebi!
    __syncthreads();

    //**********shared memory summation stage***********
    for (int length = BLOCK_SIZE / 2; length >= 1; length /= 2)
    {
        int double_kill = -1;
	if (threadIdx.x < length)
	{
	    double_kill = bowman[threadIdx.x] + bowman[threadIdx.x + length];
	}
	__syncthreads();  //why we need two __syncthreads() here, and,
	
	if (threadIdx.x < length)
	{
	    bowman[threadIdx.x] = double_kill;
	}
	__syncthreads();  //....here ?
	
    } //the per-block partial sum is bowman[0]

    if (blockDim.x * blockIdx.x < count) //in case that our users are naughty
    {
        //per-block result written back, by thread 0, on behalf of a block.
        if (threadIdx.x == 0) output[blockIdx.x] = bowman[0];
    }
}

int _hawk_sum_cpu(int *ptr, int count)
{
    int sum = 0;
    for (int i = 0; i < count; i++)
    {
        sum += ptr[i];
    }
    return sum;
}

void _nanana_init(int *ptr, int count)
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the buffer with random data
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
    _hawk_sum_gpu<<<BLOCKS, BLOCK_SIZE>>>(source, N, _partial_results);
        KEN_CHECK(cudaGetLastError());  //checking for launch failures
	
    _hawk_sum_gpu<<<1, BLOCK_SIZE>>>(_partial_results, BLOCKS, final_result);
        KEN_CHECK(cudaGetLastError());  //the same
	
    KEN_CHECK(cudaDeviceSynchronize()); //checking for run-time failurs
double t1 = get_time();

    int A = final_result[0];
    fprintf(stderr, "GPU sum: %u\n", A);


    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

double t2 = get_time();
    int B = _hawk_sum_cpu(source, N);
double t3 = get_time();
    fprintf(stderr, "CPU sum: %u\n", B);

    //******The last judgement**********
    if (A == B)
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
	
