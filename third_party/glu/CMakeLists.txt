cmake_minimum_required(VERSION 3.13)
enable_language(CUDA)

set(CMAKE_CXX_FLAGS "-O3 -m64 -Wall -std=c++11")
set(CMAKE_CC_FLAGS "-O3 -m64 -Wall -std=c++11")
set(CMAKE_NVCC_FLAGS "-O3 -std=c++11 -Xcompiler")

add_subdirectory(src)