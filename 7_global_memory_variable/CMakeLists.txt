cmake_minimum_required(VERSION 3.20)
project(global_memory LANGUAGES CXX CUDA)
# enable_language(CXX)
# enable_language(CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CUDA_STANDARD 14) #指定cuda编译器使用的cuda c++版本
set(CMAKE_CUDA_STANDARD_REQUIRED on) #指定的cuda c++版本是否是必须的
set(CMAKE_CXX_STANDARD 11) #指定c++编译器的c++版本
set(CMAKE_CXX_STANDARD_REQUIRED on) #指定的c++版本是否是必须的

# set(CMAKE_CUDA_FLAGS "-G -g -O0")
# set(CMAKE_CUDA_FLAGS_DEBUG="-G -g -O0")

# 找到cuda的运行环境
find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUD_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES:${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS:${CUDA_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "Can't found CUDA")
endif()
# 添加可执行文件
add_executable(global_memory global_memory_variable.cu ../tools/common.cuh)
target_include_directories(global_memory PRIVATE ${CUDA_INCLUDE_DIRS})
# 链接库
target_link_directories(global_memory PRIVATE ${CUDA_LIBRARIES})

