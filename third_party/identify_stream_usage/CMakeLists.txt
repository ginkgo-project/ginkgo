cmake_minimum_required(VERSION 3.17)

project(identify_stream_usage LANGUAGES CXX)
if(GINKGO_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
    add_library(identify_stream_usage_cuda SHARED identify_stream_usage.cpp)
    set_target_properties(
        identify_stream_usage_cuda
        PROPERTIES CUDA_RUNTIME_LIBRARY SHARED
    )
    target_link_libraries(
        identify_stream_usage_cuda
        PUBLIC CUDA::cudart ${CMAKE_DL_LIBS}
    )
    set_target_properties(
        identify_stream_usage_cuda
        # set target compile options
        PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
    )

    add_executable(
        test_stream_identification_cuda
        test_default_stream_identification.cu
    )
    add_test(
        NAME default_stream_identification_cuda
        COMMAND test_stream_identification_cuda
    )

    set_tests_properties(
        default_stream_identification_cuda
        PROPERTIES
            ENVIRONMENT LD_PRELOAD=$<TARGET_FILE:identify_stream_usage_cuda>
    )
endif()

if(GINKGO_BUILD_HIP AND GINKGO_HIP_PLATFORM_AMD)
    find_package(hip REQUIRED)
    set_source_files_properties(
        identify_stream_usage.hip.cpp
        test_default_stream_identification.hip.cpp
        PROPERTIES LANGUAGE HIP
    )
    add_library(identify_stream_usage_hip SHARED identify_stream_usage.hip.cpp)
    target_link_libraries(identify_stream_usage_hip ${CMAKE_DL_LIBS})
    set_target_properties(
        identify_stream_usage_hip
        # set target compile options
        PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            POSITION_INDEPENDENT_CODE ON
    )

    add_executable(
        test_stream_identification_hip
        test_default_stream_identification.hip.cpp
    )
    add_test(
        NAME default_stream_identification_hip
        COMMAND test_stream_identification_hip
    )

    set_tests_properties(
        default_stream_identification_hip
        PROPERTIES
            ENVIRONMENT LD_PRELOAD=$<TARGET_FILE:identify_stream_usage_hip>
    )
endif()
