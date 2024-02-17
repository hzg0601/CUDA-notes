#pragma once
#include <stdio.h>
#include <stdlib.h>

void set_device(int device_id=0){

    // get device_count
    int device_count = 0;
    cudaError_t error;
    error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0){
        printf("No CUDA compatiable GPU found \n");
        exit(-1);
    }
    printf("the device_count is %d\n",device_count);
    // set devices
    if (device_count < device_id+1) {
        printf("device_id is unavailable\n");
        exit(-1);
    }
    error = cudaSetDevice(device_id);
    if (error != cudaSuccess){
        printf("fail to set GPU \n");
    }
    printf("the device is set as: %d\n", device_id);

}

cudaError_t cuda_error_check(cudaError_t error_code, const char *filename, int line_number){
    if (error_code != cudaSuccess){
        printf("cuda error occured:\r\n code=%d, name=%s, description=%s\r\n file=%s,line%d",
        error_code,cudaGetErrorName(error_code),cudaGetErrorString(error_code),
        filename,line_number);
        return error_code;
    }
    return error_code;
}

void set_device_by_arg(int argc, char *argv[]){
    int devcie_id = 0;
    if (argc > 1){
        devcie_id = atoi(argv[1]);
    }
    set_device(devcie_id);
}

