
# cmake needs this line
cmake_minimum_required(VERSION 3.13.2)
 
# Define project name
project(cudnn_sample LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED on)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
# # 将指定目录添加到编译器的头文件搜索路径之下，指定的目录被解释成当前源码路径的相对路径。
include_directories(/usr/local/include/opencv4/)
include_directories(/usr/local/cuda-12.1/targets/x86_64-linux/include/)
if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    config: ${OpenCV_DIR}")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    libraries: ${OpenCV_LIBS}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else()
    message(FATAL_ERROR "Cannot found CUDA")

endif()

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
    message(STATUS "CUD_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES:${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS:${CUDA_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "Can't found CUDA")
endif()       

# Declare the executable target built from your sources
add_executable(cudnn.out cudnn_sample.cu)
 
# Link your application with OpenCV libraries
# 为指定目标（target）添加搜索路径，
# 指定目标是指通过如add_executable()，add_library()这样的命令生成的，
# 并且决不能是alias target（引用目标，别名目标）。
target_link_libraries(cudnn.out PRIVATE ${OpenCV_LIBS})
target_link_libraries(cudnn.out PRIVATE ${CUDA_LIBRARIES})
target_include_directories(cudnn.out PRIVATE ${CUDA_INCLUDE_DIRS})
