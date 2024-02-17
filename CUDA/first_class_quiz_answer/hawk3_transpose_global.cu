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

#define M 3001  //three thousand and one nights

__managed__ int shark[M][M];      //input matrix
__managed__ int gpu_shark_T[M][M];//GPU result
__managed__ int cpu_shark_T[M][M];//CPU result


__global__ void _jvav_transpose(int A[M][M], int B[M][M])
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < M && y < M)
    {
	//you cannot do this if the transpose is in-place.
	    B[y][x] = A[x][y]; //naive transpose on global memory. 
    }			
}

void _sparks_transpose_cpu(int A[M][M], int B[M][M])
{
    for (int j = 0; j < M; j++)
    {
	    for (int i = 0; i < M; i++)
	    {
	        B[i][j] = A[j][i];
	    }
    }
}

void DDBDDH_init(int A[M][M])
{
    uint32_t seed = (uint32_t)time(NULL); //make huan happy
    srand(seed);  //reseeding the random generator

    //filling the matrix with random data
    for (int j = 0; j < M; j++)
    {
	for (int i = 0; i < M; i++)
	{
	    A[j][i] = rand();
	}
    }
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
    fprintf(stderr, "DDBDDH is filling the %dx%d maxtrix with random data\n",
	            M, M);
    DDBDDH_init(shark);

    //**********************************
    //Now we are going to kick start your kernel.
    cudaDeviceSynchronize(); //steady! ready! go!
    //Good luck & have fun!
    
    fprintf(stderr, "Running on GPU...\n");
    
    double t0 = get_time();
    int n = (M + 15) / 16; //what the hell is this!
    dim3 grid_shape(n, n);
    dim3 block_shape(16, 16);
    _jvav_transpose<<<grid_shape, block_shape>>>(shark, gpu_shark_T);
    KEN_CHECK(cudaGetLastError());  //checking for launch failures
    KEN_CHECK(cudaDeviceSynchronize()); //checking for run-time failurs
    double t1 = get_time();

    //**********************************
    //Now we are going to exercise your CPU...
    fprintf(stderr, "Running on CPU...\n");

    double t2 = get_time();
    _sparks_transpose_cpu(shark, cpu_shark_T);
    double t3 = get_time();

    //******The last judgement**********
    for (int j = 0; j < M; j++)
    {
	    for (int i = 0; i < M; i++)
	    {
	        if (gpu_shark_T[j][i] != cpu_shark_T[j][i])
	        {
	            fprintf(stderr, "Test failed!\n");
	   	        exit(-1);
	        }
	    }
    }	
    fprintf(stderr, "Test Passed!\n");
    
    //****and some timing details*******
    fprintf(stderr, "GPU time %.3f ms\n", (t1 - t0) * 1000.0);
    fprintf(stderr, "CPU time %.3f ms\n", (t3 - t2) * 1000.0);

    return 0;
}	
	
