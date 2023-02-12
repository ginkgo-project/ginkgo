cmake_minimum_required(VERSION 3.17)

project(identify_stream_usage LANGUAGES CXX CUDA)

set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
add_library(identify_stream_usage SHARED identify_stream_usage.cpp)

find_package(CUDAToolkit REQUIRED)

set_target_properties(identify_stream_usage PROPERTIES CUDA_RUNTIME_LIBRARY SHARED)
target_link_libraries(identify_stream_usage PUBLIC CUDA::cudart)

set_target_properties(
  identify_stream_usage
  PROPERTIES # set target compile options
             CXX_STANDARD 17
             CXX_STANDARD_REQUIRED ON
             POSITION_INDEPENDENT_CODE ON
)

add_executable(test_stream_identification test_default_stream_identification.cu)
add_test(NAME default_stream_identification COMMAND test_stream_identification)

set_tests_properties(
  default_stream_identification PROPERTIES ENVIRONMENT
                                           LD_PRELOAD=$<TARGET_FILE:identify_stream_usage>
)