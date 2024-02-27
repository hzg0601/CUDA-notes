#include <cudnn.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
// 使用Tensor核的两个CUDA库是cuBLAS和cuDNN。
// cuBLAS使用张量核加速GEMM计算（GEMM是矩阵-矩阵乘法的BLAS术语）；cuDNN使用张量核加速卷积和递归神经网络（RNNs）。
// 在 CUDA 9.0 中，访问 Tensor Core 需要的函数和类型都定义在 nvcuda::wmma 命名空间中，这些函数允许您可以初始化 
// Tensor Core 所需的特殊格式、将矩阵加载到寄存器，执行矩阵乘加（MMA），把矩阵存回内存
// https://zhuanlan.zhihu.com/p/673397713
// WMMA API 包含在 mma.h 头文件中。完整的命名空间是 nvcuda::wmma::*，
// 但最好在整个代码中都明确使用 wmma。

//卷积核初始化
float3 data_kernel[] = {
    make_float3(-1.0f,-1.0f,-1.0f),   make_float3(0.0f,0.0f,0.0f),   make_float3(1.0f,1.0f,1.0f),
    make_float3(-2.0f,-2.0f,-2.0f),   make_float3(0.0f,0.0f,0.0f),   make_float3(2.0f,2.0f,2.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-2.0f, -2.0f, -2.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(2.0f, 2.0f, 2.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f),
    make_float3(-2.0f, -2.0f, -2.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(2.0f, 2.0f, 2.0f),
    make_float3(-1.0f, -1.0f, -1.0f), make_float3(0.0f, 0.0f, 0.0f), make_float3(1.0f, 1.0f, 1.0f)
};    

int main(){
    //读取数据，获取图片数据
    Mat img = imread("1.jpg"); 
    int img_width = img.cols;
    int img_height = img.rows;
    // channels由函数获得，不是属性
    int img_channel = img.channels();
    // 定义计算完毕的图片在host的容器用于回传数据，图片类型为Mat，通道由函数计算？
    Mat dst_gpu(img_width,img_height,CV_8UC3,Scalar(0,0,0));
    // CV_8UC3, bit_depth[s,u,f]channel,每个像素占bith_depth bite的字节，
    // 以[s,u,f]类型形式表示，
    // Signed int, Unsigned int, Float
    size_t num = img_channel * img_height * img_width * sizeof(unsigned char);

    // 声明并定义gpu上的输入，输出、卷积核，分配显存
    unsigned char * in_gpu, * out_gpu;
    float * filt_data; // 卷积核的大小
    //分配显存
    cudaMalloc((void **)&in_gpu, num);
    cudaMalloc((void **)&out_gpu, num);
    cudaMalloc((void **)&filt_data, 3*3*3*sizeof(float3));

    //初始化句柄
    // 句柄（handle）是C++程序设计中经常提及的一个术语。
    //它并不是一种具体的、固定不变的数据类型或实体，而是代表了程序设计中的一个广义的概念。
    //句柄一般是指获取另一个对象的方法——一个广义的指针，它的具体形式可能是一个整数、
    //一个对象或就是一个真实的指针，
    //而它的目的就是建立起与被访问对象之间的唯一的联系
    // 之所以要设立句柄，根本上源于内存管理机制的问题，即虚拟地址。
    //简而言之数据的地址需要变动，变动以后就需要有人来记录、管理变动，
    //因此系统用句柄来记载数据地址的变更。在程序设计中，句柄是一种特殊的智能指针，
    //当一个应用程序要引用其他系统（如数据库、操作系统）所管理的内存块或对象时，就要使用句柄
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    // 声明、创建、初始化 input,output的描述符
    cudnnTensorDescriptor_t input_descriptor;
    cudnnCreateTensorDescriptor(&input_descriptor);
    cudnnSetTensor4dDescriptor(input_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,1,3,img_height,img_width);

    cudnnTensorDescriptor_t output_descriptor;
    cudnnCreateTensorDescriptor(&output_descriptor);
    cudnnSetTensor4dDescriptor(output_descriptor,CUDNN_TENSOR_NHWC,CUDNN_DATA_FLOAT,1,3,img_height,img_width);
    // 声明、创建、初始化 卷积核的描述符
    cudnnFilterDescriptor_t kernel_descriptor;
    cudnnCreateFilterDescriptor(&kernel_descriptor);
    cudnnSetFilter4dDescriptor(kernel_descriptor,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,3,3,3,3);
    // 声明、创建、初始化卷积操作描述符
    cudnnConvolutionDescriptor_t conv_descriptor;
    cudnnCreateConvolutionDescriptor(&conv_descriptor);
    cudnnSetConvolution2dDescriptor(conv_descriptor,1,1,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT);
    // 声明、创建、初始化算法，非操作符
    cudnnConvolutionFwdAlgoPerf_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(handle,input_descriptor,kernel_descriptor,conv_descriptor,output_descriptor,1,0,&algo);
    
    //计算workspace_size的大小
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle,input_descriptor,kernel_descriptor,conv_descriptor,output_descriptor,algo.algo,&workspace_size);
    // 申请相应的空间
    void * workspace = nullptr;
    cudaMalloc(&workspace,workspace_size);
    
    //拷贝数据
    cudaMemcpy((void *)filt_data,(void *)data_kernel,3*3*3*sizeof(float3),cudaMemcpyHostToDevice);
    cudaMemcpy((void *)in_gpu, (void *)img.data, num, cudaMemcpyHostToDevice);
    
    //执行操作
    auto alpha=1.0f, beta=0.0f;
    cudnnConvolutionForward(handle,&alpha,input_descriptor,in_gpu,kernel_descriptor,filt_data,conv_descriptor,algo.algo,workspace,workspace_size,&beta,output_descriptor,out_gpu);

    //回传数据
    cudaMemcpy(dst_gpu.data,out_gpu,num,cudaMemcpyDeviceToHost);
    //释放资源
    cudaFree(in_gpu);
    cudaFree(out_gpu);
    cudaFree(filt_data);
    cudaFree(workspace);
    //销毁描述符
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyConvolutionDescriptor(conv_descriptor);
    //销毁handle
    cudnnDestroy(handle);
    //显示结果
    imshow("cudnn_sample",dst_gpu);
    // cv2 waitKey(delay),用于指定图像显示的延迟
    // 这个函数是HighGUI窗口中唯一的获取和处理事件的方法，因此它必须存在。
    return 0;

}