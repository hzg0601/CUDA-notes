#include <stdio.h>
#include <stdlib.h>
#include "../tools/common.cuh"

int get_cuda_core_info(cudaDeviceProp dev_prop){
    int num_cores = 0;
    int mp = dev_prop.multiProcessorCount;
    switch (dev_prop.major)
    {
    case 2: //Fermi
        if (dev_prop.minor == 1) num_cores = mp * 48;
        else num_cores = mp*32;
        break;
    case 3: //Kepler
        num_cores = mp * 192;
        break;
    case 5: //Pascal
        num_cores = mp * 128;
    case 6:
        if ((dev_prop.minor == 1) || (dev_prop.minor == 2)) num_cores = mp * 128;
        else if ((dev_prop.minor == 0)) num_cores = mp * 64;
        else printf("Unknown device type \n");
        break; 
    case 7: //Volta and Turing
        if ((dev_prop.minor) == 0 || (dev_prop.minor == 5)) num_cores = mp * 64;
        else printf("Unknown device type \n");
        break;
    case 8: // Ampere, ada lovelace
        if (dev_prop.minor == 0) num_cores = mp * 64;
        else if (dev_prop.minor == 6) num_cores = mp *128;
        else if (dev_prop.minor == 9) num_cores = mp * 128; //ada lovelace
        break;
    case 9://Hopper
        if (dev_prop.minor == 9) num_cores = mp * 128;  
        break;
        
    default:
        printf("Unknown device type \n");
        break;
    }
    printf("the num of cuda cores of compute capbility %d.%d are: %d\n", dev_prop.major,dev_prop.minor,num_cores);
    return num_cores;
}

int get_gpu_detail(cudaDeviceProp dev_prop){
    printf("device name:%s\n",dev_prop.name);
    printf("compute capability: %d.%d\n",dev_prop.major,dev_prop.minor);
    printf("global memory: %.2f GB\n",dev_prop.totalGlobalMem/(1024.0*1024*1024));
    printf("constant memory: %2.f KB\n", dev_prop.totalConstMem/1024.0);
    printf("maximum grid size: %d,%d,%d\n",dev_prop.maxGridSize[0],dev_prop.maxGridSize[1],dev_prop.maxGridSize[2]);
    printf("maximum block size: %d,%d,%d\n",dev_prop.maxThreadsDim[0],dev_prop.maxThreadsDim[1],dev_prop.maxThreadsDim[2]);
    printf("number of sms: %d\n",dev_prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n",
        dev_prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n",
        dev_prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:     %d K\n",
        dev_prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:        %d K\n",
        dev_prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n",
        dev_prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
        dev_prop.maxThreadsPerMultiProcessor); 
    return 0;
}

int main(int argc, char *argv[]){

    int device_id = 0;
    if (argc > 1){
        printf("the arg is %s\n",argv[1]);
        device_id = atoi(argv[1]);
        printf("the device is %d\n",device_id);
    }

    set_device(device_id);

    cudaDeviceProp dev_prop;
    cuda_error_check(cudaGetDeviceProperties(&dev_prop, device_id),__FILE__,__LINE__);
    get_cuda_core_info(dev_prop);
    get_gpu_detail(dev_prop);
    printf("done.\n");
    return 0;
}