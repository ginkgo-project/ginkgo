# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.17)

project(identify_stream_usage LANGUAGES CXX)
if(GINKGO_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
    add_library(identify_stream_usage_cuda SHARED identify_stream_usage.cpp)
    set_target_properties(identify_stream_usage_cuda PROPERTIES CUDA_RUNTIME_LIBRARY SHARED)
    target_link_libraries(identify_stream_usage_cuda PUBLIC CUDA::cudart ${CMAKE_DL_LIBS})
    set_target_properties(
        identify_stream_usage_cuda
        PROPERTIES # set target compile options
                  CXX_STANDARD 17
                  CXX_STANDARD_REQUIRED ON
                  POSITION_INDEPENDENT_CODE ON
    )
  
    add_executable(test_stream_identification_cuda test_default_stream_identification.cu)
    add_test(NAME default_stream_identification_cuda COMMAND test_stream_identification_cuda)
    
    set_tests_properties(default_stream_identification_cuda PROPERTIES
                         ENVIRONMENT LD_PRELOAD=$<TARGET_FILE:identify_stream_usage_cuda>)
endif()

if(GINKGO_BUILD_HIP AND GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    find_package(hip REQUIRED)
    set_source_files_properties(identify_stream_usage.hip.cpp
                                test_default_stream_identification.hip.cpp
                                PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    hip_add_library(identify_stream_usage_hip
                    identify_stream_usage.hip.cpp
                    HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
                    CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
                    NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS}
                    SHARED)
    target_link_libraries(identify_stream_usage_hip PUBLIC hip::amdhip64 ${CMAKE_DL_LIBS})
  
    hip_add_executable(test_stream_identification_hip
                       test_default_stream_identification.hip.cpp
                       HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
                       NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS}
                       CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS})
    add_test(NAME default_stream_identification_hip COMMAND test_stream_identification_hip)
    
    set_tests_properties(default_stream_identification_hip PROPERTIES
                         ENVIRONMENT LD_PRELOAD=$<TARGET_FILE:identify_stream_usage_hip>)
endif()